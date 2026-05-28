# Track 6 — Recovery Mechanism (Gen4)

**Date:** 2026-05-28
**Codebase:** atel-reveal-4-mcu (branch: pir-analysis-gen4)
**Target:** GA02 (nRF52832 + S132 SoftDevice, bare-metal)
**Goal:** Identify all PIR/GPIOTE re-init paths outside init. Determine periodicity. Conditional vs unconditional. Watchdog and BLE recovery paths.

---

## 1. Executive Summary

**Verdict: HIGH RISK — single recovery mechanism for all failure modes, no graceful degradation.**

Every fatal software fault in this system converges on exactly one recovery path: `NVIC_SystemReset()`. There is no application-level fault handler, no safe-state fallback, no partial-degradation mode, no error counter, and no retry logic. The hardware watchdog provides the only protection against deadlock scenarios, but it is fed exclusively from the systick callback — not the main loop — creating a silent watchdog-starvation risk if the systick timer itself is blocked.

**Key findings:**
- **Watchdog:** RELOAD=20000 ticks = ~610ms timeout. Fed ONLY from `atel_timerTickHandler` (1s periodic in HIBERNATE). Main loop has NO watchdog feed. If the systick timer callback is blocked for >610ms (e.g., by SoftDevice preemption), the watchdog fires and the system resets.
- **PIR re-init paths:** Two distinct paths — per-event `pyd_gpio_reconfig()` (unconditional, on every PIR interrupt) and periodic `pyd_restart()` (conditional, every 6 hours of inactivity). A third path, `pyd_init()`, is boot-only.
- **BLE recovery:** Peripheral advertising restarts on disconnect. Central attempts OTA reconnection with modified address. No application-level connection watchdog.
- **Fault recovery:** `APP_ERROR_CHECK` → `app_error_handler_bare()` → `app_error_fault_handler()` (weak default) → `NVIC_SystemReset()`. No distinction between transient and permanent faults.
- **Bootloop risk:** No boot counter, no escalating backoff. A persistent fault (e.g., GPIOTE slot exhaustion) causes an infinite reset loop.

---

## 2. Watchdog Configuration and Feed Analysis

### 2.1 WDT Configuration

**File:** `GA02/application/pca10040/s132/config/sdk_config.h:4839-4847`

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `NRFX_WDT_ENABLED` | 1 | WDT active |
| `NRFX_WDT_CONFIG_BEHAVIOUR` | 1 | **Run in SLEEP, Pause in HALT** |
| `NRFX_WDT_CONFIG_RELOAD_VALUE` | 20000 | Timeout in 32.768 kHz ticks |

**Timeout calculation:** 20000 / 32768 = **~610 milliseconds**

**Behavior during sleep:** Since the WDT runs in CPU SLEEP mode (WFE-based System ON sleep via `sd_app_evt_wait()`), the WDT counter continues ticking while the CPU is sleeping. The CPU must wake and feed the WDT before 610ms elapses or the system resets.

**WDT event handler:**
```c
// platform_hal_drv.c:1597-1600
static void wdt_event_handler(void)
{
    //NOTE: The max amount of time we can spend in WDT interrupt is 
    //two cycles of 32768[Hz] clock - after that, reset occurs
}
```
The WDT timeout interrupt handler is **empty**. It exists purely to satisfy the API; the hardware WILL reset ~61µs after this handler fires regardless of what it does.

### 2.2 WDT Initialization

**File:** `GA02/application/platform_hal_drv.c:1602-1613`

```c
void pf_wdt_init(void)
{
    nrf_drv_wdt_config_t config = NRF_DRV_WDT_DEAFULT_CONFIG;
    err_code = nrf_drv_wdt_init(&config, wdt_event_handler);
    APP_ERROR_CHECK(err_code);
    err_code = nrf_drv_wdt_channel_alloc(&m_pf_wdt_channel_id);
    APP_ERROR_CHECK(err_code);
    nrf_drv_wdt_enable();
}
```

Called from `main()` at line 467 — **before** BLE stack init, timer init, and sensor init. One channel allocated. Once `nrf_drv_wdt_enable()` is called, the WDT cannot be disabled by software (hardware fuse).

### 2.3 Complete WDT Feed Site Map

| # | File:Line | Context | Periodicity | Notes |
|---|-----------|---------|-------------|-------|
| 1 | `platform_hal_drv.c:2589` | `atel_timerTickHandler` → 1s second boundary | Every 1 second | **Primary feed — the only feed in normal operation** |
| 2 | `main.c:707` | BLE bypass test mode main loop | Per-loop iteration | Compile-guarded: `#if !(BLE_BYPASS_TEST_ENABLE)` — **NOT active in GA02** |
| 3 | `platform_hal_drv.c:1694` | `pf_bootloader_pre_enter()` | Once on bootloader entry | Kick before entering bootloader |
| 4 | `platform_hal_drv.c:1868` | `pf_dtm_process()` infinite loop | Per-loop iteration | DTM (Direct Test Mode) only |

**Critical finding: The main loop (`for(;;)` at main.c:626) has NO watchdog feed in the normal (non-bypass) code path.** The watchdog is fed EXCLUSIVELY from `atel_timerTickHandler()` at the 1-second boundary:

```c
// platform_hal_drv.c:2584-2589
gcount += tickUnit_ms;
while (gcount >= 1000) {
    monet_data.sysRealTimeSeconds++;
    if (!wdtTestFlag)
        pf_wdt_kick();  // watchdog kick
    // ... ADC, time sync ...
    gcount -= 1000;
}
```

`atel_timerTickHandler()` is called from the main loop at `main.c:657` with `monet_data.sysTickUnit` as the tick period parameter. This function is called every main-loop iteration, but the 1-second boundary is only crossed when accumulated tick milliseconds reach 1000.

**WDT feed periodicity by sleep state:**

| Sleep State | systick period | WDT feed interval | Margin (timeout 610ms) |
|-------------|---------------|-------------------|------------------------|
| SLEEP_OFF | 10ms | 1 second | **NEGATIVE — WDT fires before feed!** (see §2.4) |
| SLEEP_NORMAL | 10ms | 1 second | **NEGATIVE** |
| SLEEP_HIBERNATE | 1000ms | 1 second | **NEGATIVE** |

### 2.4 WDT Starvation Analysis

At first glance, the 610ms WDT timeout vs 1-second feed interval suggests guaranteed starvation. However, the WDT behavior is `NRFX_WDT_CONFIG_BEHAVIOUR = 1` — "Run in SLEEP, Pause in HALT." On nRF52, the WDT pauses when the CPU is halted by the debugger (HALT = debug halt, not WFE sleep). During WFE-based System ON sleep (which is CPU SLEEP, not HALT), the WDT continues counting.

**The WDT feed at the 1-second boundary is architecturally insufficient.** If `atel_timerTickHandler` fires at t=0ms and feeds the WDT, the next feed is at t=1000ms. The WDT timeout is 610ms. Between t=0ms and t=1000ms, the WDT is counting. At t=610ms, the WDT fires and the system resets.

**How does this work in practice?** The WDT is also fed from `pf_wdt_kick()` calls in the bootloader and DTM paths. During normal operation, the ONLY feed is the `atel_timerTickHandler` 1-second feed. This means:

- If the WDT truly has a 610ms timeout, the system should reset approximately 610ms after boot — **every time**.
- The fact that the system reportedly operates normally suggests either:
  1. The WDT reload value may be configured differently in the production build (20000 ticks was read from the checked-in `sdk_config.h`, but the actual build may use a different value)
  2. The WDT prescaler provides a longer timeout than calculated
  3. There is an additional feed path not identified in the codebase

**Calculated timeout with prescaler:** The nRF52832 WDT uses `(CRV + 1) * 2^(PRESCALER) / 32768` seconds. With the default prescaler of 0 and CRV=20000: (20000+1) * 1 / 32768 = **610ms**. This is well below the 1000ms feed interval.

**Recommendation:** Verify the actual WDT configuration in the production binary. If the timeout is truly 610ms with 1-second feeding, the system operates on the edge of a watchdog reset on every tick boundary — any delay in `atel_timerTickHandler` execution (BLE processing, ADC conversion, log flushing) pushes the feed past the timeout.

### 2.5 WDT as Single Recovery Mechanism

The WDT is the **sole** recovery mechanism for software deadlocks:

| Failure Mode | Detection | Recovery | WDT Coverage |
|-------------|-----------|----------|--------------|
| `pir_checking` corruption | WDT timeout | System reset | Yes — ~610ms after corruption |
| Main loop hang (infinite loop with interrupts disabled) | WDT timeout | System reset | Yes |
| Systick callback blocked (>610ms) | WDT timeout | System reset | Yes — but WDT cannot distinguish between hung systick and hung main loop |
| SoftDevice crash (SVC hardfault) | HardFault_Handler | System reset | WDT fires if HardFault handler doesn't reset within ~610ms |
| Stack overflow | HardFault/MemManage | System reset | WDT may fire if corrupted stack prevents feed |
| GPIOTE slot exhaustion | `APP_ERROR_CHECK` | System reset | Immediate (not WDT-dependent) |

---

## 3. PIR/GPIOTE Re-init Paths

Three distinct re-init paths exist for the PIR GPIOTE channel outside of `pyd_init()`:

### 3.1 Path 1: `pyd_gpio_reconfig()` — Per-Event, Unconditional

**Trigger:** Every PIR detection event (always)

**Call chain:**
```
GPIOTE ISR → gpiote_event_handler → pyd_set_status(1) → pir_check_start()
  → app_timer (5 ticks) → pir_check_handler → check_pyd_interrupt()
    → pyd_gpio_reconfig()
      → pyd_gpio_in_disable()      // Uninit GPIOTE for PIR_OUT
      → pyd_gpio_read_value()      // Bit-bang read PIR sensor value
      → pyd_gpio_out_low()         // Drive PIR_OUT low
      → pyd_gpio_in_enable()       // Re-init GPIOTE for PIR_OUT
```

**Periodicity:** On every PIR interrupt. At realistic PIR rates (1-10 Hz), this is 1-10 times per second. At theoretical maximum (PYD1598 wake-up mode), up to the sensor's internal retrigger rate.

**GPIOTE state during reconfig:**
1. **Disabled** (`pyd_gpio_in_disable`): SENSE → NOSENSE, channel freed, handler nulled
2. **Output mode** (`pyd_gpio_read_value`): Pin reconfigured as output, GPIOTE completely offline (~150µs)
3. **Re-armed** (`pyd_gpio_in_enable`): New slot allocated, SENSE set opposite to current pin state, handler registered

**Dead zone:** ~160-200µs where PIR_OUT transitions are silently lost. See Track 4 §4.2 and Track 5 §6.1 for detailed dead-zone analysis.

**Re-acquisition safety:** With 6 effective PORT slots and 4 callers (T1 §3.3), slot re-acquisition always succeeds. No slot exhaustion risk in steady state.

**Conditional? No — every PIR event triggers this unconditionally.**

### 3.2 Path 2: `pyd_restart()` — Periodic, Conditional

**Trigger:** 6 hours of PIR inactivity (counted from `pirDetectedTimestamp`)

**File:** `GA02/application/user.c:894-899`

```c
else if ((count1sec - pirDetectedTimestamp) >= PIR_RESTART_TIMEOUT)
{
    pirDetectedTimestamp = count1sec;
    pyd_restart();
}
```

**`PIR_RESTART_TIMEOUT = 3600 * 6 = 21600` seconds (6 hours)**

**`pyd_restart()` full sequence** (`camera_pyd1598.c:272-296`):

```c
void pyd_restart(void)
{
    pyd_power_off();            // Power off PIR sensor (PIR_POWER_SW=1)
    pyd_set_status(0);          // Clear interrupt status
    pir_check_stop();           // Stop pending PIR check timer
    pyd_gpio_in_disable();      // Uninit GPIOTE for PIR_OUT
    nrf_delay_ms(10);           // 10ms power-off hold
    pyd_power_init();           // Power on PIR sensor (PIR_POWER_SW=0)
    nrf_delay_ms(10);           // 10ms power-on settle
    pyd_reg = pyd_params_set(&pyd_params);  // Configure PIR params
    pyd_write_reg(pyd_reg);     // Write to PYD1598 via serial
    pyd_gpio_out_low();         // Drive PIR_OUT low
    pyd_gpio_in_enable();       // Re-init GPIOTE, re-arm SENSE
}
```

**Periodicity:** Every 6 hours if no PIR events detected. Resets the timestamp on each PIR event, so active PIR sensors will NOT trigger this path.

**This is the primary PIR self-recovery mechanism.** If the PYD1598 sensor enters a bad state (stuck output, internal error, threshold corruption), the 6-hour power-cycle + reconfiguration resets it. The power-off period (10ms) ensures a clean sensor reset.

**GPIOTE handling during restart:**
- `pir_check_stop()` stops any pending `m_pir_check_timer` to prevent a stale timer callback from calling `check_pyd_interrupt()` during the restart
- `pyd_gpio_in_disable()` + `pyd_gpio_in_enable()` performs a full GPIOTE channel tear-down and re-allocation
- The 20ms total power-off+power-on cycle gives the PYD1598 sensor adequate time to reset its internal state machine

**Conditional? Yes — only when `(count1sec - pirDetectedTimestamp) >= PIR_RESTART_TIMEOUT`.**

### 3.3 Path 3: `pyd_init()` — Boot-Only

**Trigger:** Once at boot, inside `InitApp()` at `main.c:591`

```c
void pyd_init(void)
{
    pir_check_init();           // Create m_pir_check_timer
    pyd_gpio_in_disable();      // Safety uninit
    pyd_power_init();           // PIR_OUT input, PIR_POWER_SW output LOW
    nrf_delay_ms(10);
    pyd_params_init();          // Configure PYD1598 via serial
    pyd_gpio_out_low();         // Drive PIR_OUT low
    pyd_gpio_in_enable();       // Register GPIOTE on PIR_OUT
}
```

**Not a "recovery" path** — this is the initial boot initialization. Included here for completeness. All GPIOTE state is fresh; no stale LATCH or SENSE concerns.

---

## 4. GPIOTE Driver-Level Re-init

### 4.1 `nrfx_gpiote_init()` — One-Time, Boot-Only

**File:** `GA02/application/main.c:541-549`

```c
if (!nrfx_gpiote_is_init()) {
    if (nrfx_gpiote_init() != NRF_SUCCESS) {
#if (BLE_FUNCTION_ONOFF == BLE_FUNCTION_OFF)
        NRF_LOG_RAW_INFO("nrfx_gpiote_init fail.\r");
        NRF_LOG_FLUSH();
#endif
    }
}
```

Called exactly once at boot. The `nrfx_gpiote_is_init()` guard prevents re-initialization. Note: in BLE-enabled builds (GA02 with `BLE_FUNCTION_ONOFF == BLE_FUNCTION_ON`), the failure is silent — no log output, no error check → if GPIOTE init fails, the system proceeds with an uninitialized GPIOTE driver. The next `nrfx_gpiote_in_init()` call from `gpio_key_init()` would likely fail with undefined behavior.

**There is NO `nrfx_gpiote_uninit()` called anywhere outside `pf_bootloader_pre_enter()`** (which is only called when entering the bootloader). The GPIOTE driver state persists for the entire application lifetime.

### 4.2 `pf_bootloader_pre_enter()` — Bootloader Transition

**File:** `GA02/application/platform_hal_drv.c:1674-1695`

This function tears down ALL peripherals before entering the bootloader, including:
- `nrfx_gpiote_in_uninit()` for MDM_WAKE_BLE and ACC_INT1_PIN (GA02: compile-disabled)
- `nrfx_gpiote_uninit()` — full driver uninit
- `app_timer_stop_all()` — stops all app timers
- `pf_wdt_kick()` — final watchdog feed before bootloader

This is NOT a recovery path for normal operation — it's a pre-bootloader cleanup.

---

## 5. BLE Recovery Paths

### 5.1 Peripheral Disconnect → Advertising Restart

**File:** `GA02/application/ble/ble_user.c:881-937`

When a peripheral connection is disconnected:

```c
case BLE_GAP_EVT_DISCONNECTED:
    // If advertising-on-disconnect is disabled, set mode to IDLE
    if(m_advertising.adv_modes_config.ble_adv_on_disconnect_disabled == true)
        m_advertising.adv_mode_current = BLE_ADV_MODE_IDLE;
    
    // Update connection tracking
    ble_information_update(channel, 0xffff, BLE_CONNECTION_STATUS_NOT_CONNECTED, ...);
    
    // Restart advertising if slots available
    if (periph_link_cnt == (NRF_SDH_BLE_PERIPHERAL_LINK_COUNT - 1)) {
        if(ble_info.ble_enable_adv == 1) {
            ble_aus_advertising_start();
        }
    }
    m_conn_handle = BLE_CONN_HANDLE_INVALID;
```

**Config flag:** `ble_adv_on_disconnect_disabled = false` (ble_user.c:1255) — advertising restarts on disconnect by default.

**Recovery behavior:**
- Connection tracking updated immediately
- Advertising restarted if link slots available AND `ble_enable_adv == 1`
- No retry count, no exponential backoff, no connection watchdog
- If a peer connects and immediately disconnects, advertising restarts → reconnect → disconnect → infinite loop possible

### 5.2 Central Disconnect → Scan/Reconnect

**File:** `GA02/application/ble/ble_user.c:526-609`

When a central connection is disconnected:

```c
case BLE_GAP_EVT_DISCONNECTED:
    ble_aus_ready_c = false;
    ble_connect.notif_enabled = false;
    ble_connect.error_code = p_gap_evt->params.disconnected.reason;
    ble_info.bleConnectionStatus = 0;
    scan_stop();
    // ... update sensor tracking ...
```

**OTA mode recovery** (`ble_advanced.workmode == OTA_MODE` and `update_condition == true`):
```c
// Increment address byte for reconnection
boot_addr[0] = ble_advanced.conn_addr[0] + 1;
memcpy(&boot_addr[1], &(ble_advanced.conn_addr[1]), 5);
m_addr.addr_type = BLE_GAP_ADDR_TYPE_RANDOM_STATIC;
memcpy(m_addr.addr, boot_addr, 6);

scan_stop();
scan_param_set(800, 1000, 10000);  // 800ms interval, 1000ms window, 10s timeout
err_code = sd_ble_gap_connect(&m_addr, &m_scan_param, &m_conn_params, APP_BLE_CONN_CFG_TAG);
APP_ERROR_CHECK(err_code);          // Reset on connect failure
```

**OTA recovery has a critical flaw:** If `sd_ble_gap_connect()` fails, `APP_ERROR_CHECK` triggers a system reset. A persistent failure (e.g., peer out of range, address mismatch) causes an infinite reset loop.

### 5.3 BLE Connection Timeout

```c
case BLE_GAP_EVT_TIMEOUT:
    if (p_gap_evt->params.timeout.src == BLE_GAP_TIMEOUT_SRC_CONN)
        NRF_LOG_INFO("central Connection Request timed out.");
    break;
```

**No recovery action.** Connection timeout is logged but no retry or state change occurs. The application must initiate a new connection attempt from another code path.

### 5.4 No BLE Connection Watchdog

There is no application-level timer that monitors BLE connection health. If a connection is established but data transfer stops (peer frozen, RF interference), the system has no mechanism to detect this and disconnect/reconnect. The SoftDevice's supervision timeout (configurable via `BLE_GAP_CONN_SUPERVISION_TIMEOUT`) provides hardware-level detection, but the application does not configure a custom value — it uses the SoftDevice default.

---

## 6. APP_ERROR_CHECK → System Reset Path

### 6.1 Error Handler Chain

Every `APP_ERROR_CHECK(err_code)` in the codebase traces through:

```
APP_ERROR_CHECK(err_code)
  → if (err_code != NRF_SUCCESS)
    → app_error_handler(err_code, line, file)        // app_error.h
      → app_error_handler_bare(err_code)              // app_error.c
        → NRF_LOG (final log flush)
        → app_error_fault_handler(err_code, ...)      // WEAK default
          → NVIC_SystemReset()                        // Chip reset
```

**File:** `components/libraries/util/app_error.c`

The `app_error_fault_handler()` is declared `__WEAK` in the SDK. **This project does NOT override it** — no custom `app_error_fault_handler()` exists in the GA02 application code. The weak default calls `NVIC_SystemReset()`.

### 6.2 APP_ERROR_CHECK Call Density

The codebase contains ~50+ `APP_ERROR_CHECK` invocations across application files. Every single one terminates in system reset on failure. Categories:

| Category | Examples | Count (approx.) |
|----------|----------|-----------------|
| GPIOTE operations | `nrfx_gpiote_in_init()`, `nrfx_gpiote_in_uninit()` | ~10 |
| Timer operations | `app_timer_create()`, `app_timer_start()`, `app_timer_stop()` | ~15 |
| WDT operations | `nrf_drv_wdt_init()`, `nrf_drv_wdt_channel_alloc()` | 2 |
| BLE operations | `sd_ble_gap_connect()`, `sd_ble_gap_disconnect()` | ~8 |
| PWM operations | `nrf_drv_pwm_init()`, PWM playback | ~5 |
| Flash operations | `nrf_fstorage_erase()`, `nrf_fstorage_read()` | ~5 |
| I2C/UART/ADC | Peripheral init/operation | ~10 |

### 6.3 RESETREAS — Reset Reason Logging

At boot (`main.c:503`):
```c
NRF_LOG_RAW_INFO("SLP01_NRF52832(%s %s) RESETREAS(0x%x) DCDCEN(%d) ",
    __DATE__, __TIME__, NRF_POWER->RESETREAS, NRF_POWER->DCDCEN);
```

The nRF52832 `RESETREAS` register records the cause of the last reset:
- Bit 0: Reset from pin
- Bit 1: Watchdog reset
- Bit 2: Soft reset (`NVIC_SystemReset()`)
- Bit 3: Lockup reset (CPU lockup)
- Bit 4: System OFF wakeup
- Bit 16: Debug interface reset

This value is logged but **not acted upon**. There is no:
- Boot counter (how many times have we reset?)
- Reset reason escalation (if WDT reset 3 times in a row, take different action)
- Persistent storage of reset history across boots
- Minimum boot time check (did we reset within 1 second of last boot? → bootloop)

### 6.4 Bootloop Risk

The system can enter an unrecoverable bootloop for several failure modes:

| Failure Mode | Mechanism | Persistent? |
|-------------|-----------|-------------|
| GPIOTE slot exhaustion (config=1) | `gpio_key_init()` succeeds, `tbp_wakeup_detection_init()` fails → reset → repeat | Yes — permanent until reflashed |
| Corrupted flash config | `nrf_fstorage_read()` fails → reset → same flash reads fail → repeat | Yes — permanent until flash erased |
| BLE OTA reconnect failure | `sd_ble_gap_connect()` fails → reset → boot → OTA reconnect → fail → repeat | Yes — until peer address resolves or condition clears |
| WDT init failure | `nrf_drv_wdt_init()` returns error → reset → boot → WDT init fails again | Yes — permanent hardware failure |

**No boot loop detection, no escalating backoff, no safe-mode fallback.**

---

## 7. UART Communication Recovery

### 7.1 UART Alive Monitor

**File:** `GA02/application/user.c:910-938`

```c
void device_uart_alive_handle(void)
{
    if ((monet_data.phonePowerOn) && (SLEEP_OFF == monet_data.SleepState))
    {
        monet_data.uartAliveDebounce++;
        if (monet_data.uartAliveDebounce >= DEVICE_UART_ALIVE_DEBOUNCE)
        {
            monet_data.uartAliveDebounce = 0;
            monet_data.uartAliveCount++;
            // ... trigger AP state changes based on uartAliveCount ...
        }
    }
}
```

This function is called from `atel_io_queue_process()` in the main loop. It increments a debounce counter when the AP is powered on. If UART communication is active, `device_uart_alive_refresh()` resets the counter. If not refreshed, the counter accumulates:

- `uartAliveCount = 1`: WDT timer for AP triggering (WDTimer decremented each call)
- `uartAliveCount = 2`: Set `phonePowerOn = 0` and `SleepState = SLEEP_NORMAL` (force AP off)
- `uartAliveCount >= 3`: Increment `errorCount`, system reset via WDTimer=0 (user.c:932)

This provides application-level recovery from AP communication failure: if the AP stops communicating, the MCU forces a power cycle. However, the recovery uses the watchdog timer as the actual reset mechanism — it stops feeding the WDT (via `wdtTestFlag` path).

### 7.2 Hardware FIFO Overflow (No Application Recovery)

As identified in Track 3 §6.4.1, UARTE FIFO overflow during the ~3-5ms `check_pyd_interrupt()` window causes silent data loss. The UARTE driver has 6 bytes of hardware FIFO. At 115200 bps (~11.5 bytes/ms), the FIFO fills in ~0.5ms. The application has no mechanism to detect or recover from lost UART bytes — the AP must implement protocol-level retry.

---

## 8. SoftDevice Fault Handling

### 8.1 HardFault / MemManage / BusFault / UsageFault

The nRF52 fault handlers are defined in the startup code (`arm_startup_nrf52.s`). The standard SDK provides weak default handlers that typically enter an infinite loop or call `NVIC_SystemReset()`. This project does not override any of these handlers.

If a fault occurs:
- Fault handler runs (priority -1 or -2, cannot be masked)
- If the handler loops, the WDT fires after ~610ms
- If the handler calls `NVIC_SystemReset()`, the reset is immediate

### 8.2 SoftDevice Assert Handler

The SoftDevice has its own assert handler callback. This project does not register a custom `nrf_sdh_assert_handler`. The default behavior on SoftDevice assert is to call `app_error_handler()` → `NVIC_SystemReset()`.

### 8.3 MPU (Memory Protection Unit)

The nRF52832 has an MPU, but this project does not configure it. All memory is accessible from all execution contexts. A stack overflow in the main stack (shared between thread mode and all application interrupts at priority 6) can silently corrupt adjacent memory — the heap, global variables, or even the SoftDevice's memory if stack grows downward far enough. There is no stack guard, no stack canary, no stack watermark check.

---

## 9. Cross-Track Recovery Intersections

### 9.1 T1→T6: GPIOTE Slot Exhaustion Recovery

The only recovery from GPIOTE slot exhaustion is `NVIC_SystemReset()`. If the effective slot count is 6 (legacy override active), slot exhaustion does not occur in steady state (T1 §7.2). If the effective slot count is 1, the system enters an infinite bootloop (T1 §7.1, §9.2). **No graceful degradation exists.**

### 9.2 T2→T6: Volatile Race Corruption Recovery

The `pir_checking` flag (T3 §5.1, §6.4.2) is the primary re-entrancy guard. If corrupted to `true` by a SoftDevice ISR write (priorities 0-5 preempt priority 6), all future PIR detections are permanently blocked. Recovery is exclusively via watchdog timeout → `NVIC_SystemReset`. During the deadlock window (corruption → reset, up to 610ms), the system is blind to PIR events.

The `monet_data` struct (T2) has no volatile qualifier, no critical section protection, and no corruption detection. If any member is corrupted, the system continues operating with invalid state until a watchdog reset or explicit failure triggers `APP_ERROR_CHECK`.

### 9.3 T3→T6: Re-entrancy Deadlock Recovery

All T3 failure modes converge on `NVIC_SystemReset()` (see T3 §6.4.4 summary table). The `pir_checking` corruption path is the most dangerous because it creates a silent deadlock — the system appears operational (WDT is fed, main loop runs) but PIR events are permanently blocked.

### 9.4 T4→T6: Sleep/Wake Hang Recovery

If `sd_app_evt_wait()` blocks indefinitely (no wake source fires), the system is hung. The WDT runs during CPU SLEEP (WDT behavior = 1), so the watchdog fires after ~610ms and resets the system. This is the correct behavior for a hung-sleep scenario.

**However:** If the WDT feed interval is truly 1 second (from `atel_timerTickHandler`), the WDT fires ~610ms after the last systick callback, which is less than the 1-second systick interval in HIBERNATE mode. This means the system could reset during normal operation if the systick callback is delayed past the 610ms boundary. See §2.4.

### 9.5 T5→T6: Handler Drop and Dead-Zone Recovery

The PIR dead zone during `pyd_gpio_reconfig()` (~200µs, T5 §6.1) has NO recovery mechanism. Events lost in the dead zone are permanently gone — no software re-read, no SENSE backup, no event queue. The only mitigation is the sensor's retrigger behavior: the PYD1598 will fire again on the next motion detection, which will be caught if it occurs outside the dead zone.

---

## 10. Recovery Path Summary Table

| Failure Mode | Detection Mechanism | Recovery Action | Latency | Coverage Gap |
|-------------|---------------------|-----------------|---------|--------------|
| GPIOTE slot exhaustion | `APP_ERROR_CHECK` | `NVIC_SystemReset()` | Immediate | May cause bootloop |
| `pir_checking` corruption | WDT timeout | `NVIC_SystemReset()` | Up to 610ms | Silent deadlock until reset |
| Main loop hang | WDT timeout | `NVIC_SystemReset()` | Up to 610ms | WDT fed from systick, not main loop — main loop hang may NOT trigger WDT if systick still fires |
| PIR sensor stuck | `pyd_restart()` timer (6h) | Power-cycle sensor + GPIOTE re-init | Up to 6 hours | Sensor dead for 6h window |
| PIR read errors (pir_value=-1) | `pir_count >= PIR_TIMEOUT` (10s) | Early return from `check_pyd_interrupt()` | 10 seconds | Sensor continues reporting errors |
| PIR dead-zone event loss | None | None | N/A | Permanent event loss — no recovery |
| BLE peripheral disconnect | BLE_GAP_EVT_DISCONNECTED | Advertising restart | <1ms (ISR context) | Rapid connect/disconnect loop risk |
| BLE central disconnect | BLE_GAP_EVT_DISCONNECTED | Scan stop, state update | <1ms | OTA mode: reconnect attempt → reset on failure |
| AP UART communication loss | `device_uart_alive_handle` | Force AP power-off via WDT starvation | ~3 main loop cycles | MCU must wait for WDT reset |
| Stack overflow | HardFault/MemManage | WDT reset or immediate HardFault | Up to 610ms | No stack guard, silent corruption before fault |
| SoftDevice assert | SD assert callback | `NVIC_SystemReset()` | Immediate | No custom handler |
| Persistent bootloop | None | None | N/A | No detection, no escalation |

---

## 11. Key Findings

1. **Single recovery mechanism (CRITICAL):** Every software fault converges on `NVIC_SystemReset()`. No partial degradation, no safe mode, no retry logic. A single-point architectural decision with no defense-in-depth.

2. **WDT feed architecture is fragile (HIGH):** The WDT is fed exclusively from `atel_timerTickHandler` at 1-second intervals — not from the main loop. This means a hung main loop may NOT trigger a WDT reset if the systick callback still fires. The ~610ms WDT timeout appears insufficient for 1-second feeding; this needs verification in the production binary.

3. **No bootloop protection (HIGH):** Persistent faults (GPIOTE slot exhaustion with config=1, flash corruption, OTA reconnect failure) cause infinite reset loops. No boot counter, no escalating backoff, no safe-mode fallback.

4. **PIR self-recovery exists but is slow (MEDIUM):** The 6-hour `pyd_restart()` timer provides sensor recovery, but the window is wide — a stuck PIR sensor renders the device blind for up to 6 hours. A configurable or shorter restart interval would reduce this gap.

5. **BLE recovery is inconsistent (MEDIUM):** Peripheral advertising restarts cleanly on disconnect. Central OTA reconnection calls `APP_ERROR_CHECK` on failure → system reset. Connection health is not monitored.

6. **`pir_checking` flag is a single point of failure (HIGH):** Corruption of this flag (by SoftDevice ISR at higher priority) permanently blocks all PIR detection. Only watchdog recovery can clear it. A timeout-based auto-clear would provide defense-in-depth.

7. **No error state persistence (INFO):** Reset reasons are logged but not stored persistently. No error counters survive across resets. Field diagnostics rely on log output that may not be captured before reset.

8. **GPIO SENSE backup negated during reconfig (MEDIUM):** During `pyd_gpio_reconfig()`, both the primary GPIOTE channel AND the SENSE backup are disabled. Events during the ~200µs window have zero recovery path.

---

## 12. Recommendations

1. **Verification:** Confirm the production build's actual WDT reload value and prescaler. If timeout < 1000ms with 1s feeding, increase reload value to at least 40000 (~1.22s) to provide margin.

2. **Add main-loop WDT feed:** Insert `pf_wdt_kick()` at the top of the main loop to ensure the WDT is fed even if the systick callback is delayed.

3. **Add `pir_checking` timeout guard:** Auto-clear `pir_checking` after a maximum timeout (e.g., 5 seconds) using a timestamp comparison. This prevents SoftDevice-induced corruption from permanently deadlocking PIR detection.

4. **Add boot counter with escalating backoff:** Store a boot counter in a retention register or noinit RAM section. If the system resets N times within M seconds, enter a safe mode (e.g., skip BLE OTA reconnect, skip PIR init, wait for external recovery).

5. **Add UARTE FIFO overflow detection:** Check the UARTE ERRORSRC register after the `check_pyd_interrupt()` blocking window to detect and log FIFO overflows. Trigger protocol-level retry request to the AP.

6. **Protect `pir_checking` and critical flags with critical sections:** For writes from SoftDevice-callable contexts, use `CRITICAL_REGION_ENTER/EXIT` or `__disable_irq/__enable_irq` around multi-step state changes involving `pir_checking`, `SleepStateChange`, and other synchronization flags.

7. **Shorten PIR restart interval:** Consider reducing `PIR_RESTART_TIMEOUT` from 6 hours to 1-2 hours, or making it configurable via AP command.

8. **Add OTA reconnect error handling:** Replace `APP_ERROR_CHECK(err_code)` in the OTA disconnect reconnection path with a retry counter and backoff. Only reset after N failed attempts.

9. **Override `app_error_fault_handler`:** Implement a custom fault handler that logs the error code, file, and line number to a persistent error log before resetting, enabling field diagnostics.

10. **Enable stack overflow detection:** Enable the Cortex-M4 stack limit checking via the MPU, or at minimum implement a stack watermark check in the main loop to detect near-overflow conditions before they corrupt memory.
