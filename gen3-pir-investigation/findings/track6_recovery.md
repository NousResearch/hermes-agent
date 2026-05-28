# Track 6: Recovery Mechanism — How Does the Device Self-Heal?

**Date:** 2026-05-28
**Firmware:** GA02-IrbisMcu (GA01 identical for PIR path)
**Hypothesis Verdict:** **CONFIRMED WITH CAVEAT** — A single coarse timer restores PIR capability every 6 hours. No active detection of lost capability exists. This is a workaround, not a root-cause fix.

---

## 1. Executive Summary

The device has exactly **one periodic recovery path** for PIR interrupt capability loss: `pyd_restart()` triggered by a 6-hour inactivity timer (`PIR_RESTART_TIMEOUT = 21600` seconds). Two additional catastrophic recovery paths exist (watchdog reset, threshold change) but neither is periodic. BLE events do NOT trigger any PIR re-initialization.

**Critical finding:** There is **no active detection** of lost PIR capability. The 6-hour `pyd_restart()` fires regardless of whether the PIR is actually working. Combined with Track 5's finding that PIR events ARE received but silently dropped at three layers, this confirms the recovery timer is a **workaround** — the developers observed PIR events "sometimes stop coming" and added a periodic full restart as a band-aid rather than fixing the root cause.

**Recovery period:** 6 hours. This constrains plausible root causes: any mechanism that requires >6 hours to manifest would leave the device permanently blind until the timer fires. Any mechanism resolved in <6 hours by this timer explains the observed "intermittent" symptom pattern.

---

## 2. Complete Map of Recovery Paths

### 2.1 Path A: `pyd_restart()` via 6-Hour Inactivity Timer (PRIMARY)

**Trigger:** `check_pyd_interrupt()` in main loop, every iteration
**Guard:** `(count1sec - pirDetectedTimestamp) >= PIR_RESTART_TIMEOUT`
**Period:** 6 hours (21600 seconds)
**Type:** Timer-driven, blind (no health check)

```
main.c:605  for(;;)
main.c:641    check_pyd_interrupt()
                → if(pyd_get_status()) ... process event
                → else if timeout → pyd_restart()   ← RECOVERY HERE
```

**Key variables:**
| Variable | Type | File:Line | Description |
|----------|------|-----------|-------------|
| `count1sec` | `volatile uint32_t` | user.c:35 | Monotonic seconds counter, incremented in `atel_timerTickHandler()` 1s path (line 1378) |
| `pirDetectedTimestamp` | `static volatile uint32_t` | user.c:43 | Last PIR event or `pyd_restart()` timestamp |
| `PIR_RESTART_TIMEOUT` | `#define 3600*6` | user.h:98 | 21600 seconds = 6 hours |

**`pirDetectedTimestamp` update points:**
1. `check_pyd_interrupt()` line 713 — on PIR event (`pyd_get_status() == 1`)
2. `check_pyd_interrupt()` line 787 — on `pyd_restart()` completion

This means the timer is effectively: "if no PIR event in the last 6 hours, restart everything." If PIR events are being silently dropped (Track 5), the timer still fires because `pirDetectedTimestamp` is only updated when events survive the drop layers and set `pyd_interrupt_status`.

**`check_pyd_interrupt()` full logic** (user.c:706-793):
```c
void check_pyd_interrupt(void) {
    static uint8_t pir_count = 0;
    pir_checking = true;
    if (pyd_get_status()) {                    // PIR event pending?
        pirDetectedTimestamp = count1sec;      // reset recovery timer
        pyd_set_status(0);
        pir_value = pyd_gpio_reconfig();       // re-register GPIOTE
        // ... process event ...
    }
    else if ((count1sec - pirDetectedTimestamp) >= PIR_RESTART_TIMEOUT) {
        pirDetectedTimestamp = count1sec;      // prevent re-entry
        pyd_restart();                         // FULL recovery
    }
    pir_checking = false;
}
```

**Evidence from firmware logs (user.c:789):**
```c
NRF_LOG_RAW_INFO("++++++++++++++++++++++++++++pyd_restart++++++++++++++++++++++++++++\n");
```
This log message confirms the restart path is instrumented and intentional — developers expected this code to execute periodically in the field.

---

### 2.2 Path B: `pyd_gpio_reconfig()` via PIR Events (NOT A RECOVERY PATH)

**This is NOT a recovery path** — it only fires after a PIR event has been successfully received and processed:

```
PIR edge → GPIOTE ISR → gpiote_event_handler (camera_pyd1598.c:167)
  → pyd_set_status(1)
  → pir_check_start() → timer → pir_check_handler → check_pyd_interrupt()
  → pyd_gpio_reconfig()
    → pyd_gpio_in_disable()    // unregister from GPIOTE
    → pyd_gpio_read_value()    // bit-bang read (~600µs)
    → pyd_gpio_out_low()
    → pyd_gpio_in_enable()     // re-register GPIOTE
```

If PIR events are being silently dropped (Track 5 layers 1-3), **this path never fires**. It cannot recover from a lost-capability state because it requires the capability to trigger.

**Dead window:** ~620µs (Track 4 §3.3). Every PIR event creates a ~620µs window where the next PIR edge is silent. This is a **self-inflicted** next-event blind spot, documented in Errata 75.

---

### 2.3 Path C: `pyd_init()` via Boot (CATASTROPHIC ONLY)

```c
// main.c:570
pyd_init();  // PIR init — one-time at boot
```

Full sequence (`camera_pyd1598.c:253-270`):
1. Initialize PIR check timer
2. Unregister from GPIOTE
3. Initialize power pin
4. 10ms delay
5. Configure PIR sensor parameters (25-bit serial)
6. Output low
7. Re-register GPIOTE

This only executes on full system reset. Not periodic.

---

### 2.4 Path D: `pyd_set_threshold()` via Config Change (RARE)

```c
// camera_pyd1598.c:60-67
void pyd_set_threshold(uint8_t num) {
    if (num >= 0 && num < 9) {
        pyd_params.threshold = pyd_threshold[num];
        pyd_restart();  // full re-init on threshold change
    }
}
```

Called from `pir_set_threshold()` (user.c:654-672) when the PIR sensitivity NV parameter changes. Not periodic — only triggered by user configuration.

---

### 2.5 Path E: Watchdog Reset (CATASTROPHIC ONLY)

**Configuration** (sdk_config.h:4830-4856):
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `NRFX_WDT_ENABLED` | 1 | Watchdog active |
| `NRFX_WDT_CONFIG_BEHAVIOUR` | 1 | Pause in CPU SLEEP mode |
| `NRFX_WDT_CONFIG_RELOAD_VALUE` | 20000 | ~610ms timeout (20000/32768) |
| `NRFX_WDT_CONFIG_NO_IRQ` | 0 | IRQ handling included |
| `NRFX_WDT_CONFIG_IRQ_PRIORITY` | 6 | Same as GPIOTE |

**Initialization** (platform_hal_drv.c:1076-1087):
```c
void pf_wdt_init(void) {
    nrf_drv_wdt_config_t config = NRF_DRV_WDT_DEAFULT_CONFIG;
    nrf_drv_wdt_init(&config, wdt_event_handler);  // handler is EMPTY
    nrf_drv_wdt_channel_alloc(&m_pf_wdt_channel_id);
    nrf_drv_wdt_enable();
}
```

**IRQ handler** (platform_hal_drv.c:1071-1074):
```c
static void wdt_event_handler(void) {
    //NOTE: The max amount of time we can spend in WDT interrupt
    //is two cycles of 32768[Hz] clock - after that, reset occurs
}
```
The handler is **empty**. The WDT fires a warning IRQ first, then resets on the second timeout. This is the standard nRF5 SDK two-stage watchdog pattern.

**Kick points:**
- `pf_systick_handler()` — every systick interrupt (1s or 10ms)
- `atel_io_queue_process()` line 2059 — main loop UART processing
- `pf_bootloader_pre_enter()` line 1168 — pre-DFU

**Forced reset via UART alive mechanism** (user.c:801-832):
```c
void device_uart_alive_handle(void) {
    if (phonePowerOn && SleepState == SLEEP_OFF) {
        monet_data.uartAliveDebounce++;
        if (uartAliveDebounce >= DEVICE_UART_ALIVE_DEBOUNCE) {  // 30s
            // UART re-init attempt
            if (uartAliveCount >= DEVICE_UART_ALIVE_COUNT_LIMIT) {  // 10 attempts
                monet_gpio.WDtimer = 0;  // WDTimer == 0 → system reset
            }
        }
    }
}
```
After 10 consecutive UART failures (~5 minutes of UART silence), `WDtimer` is zeroed, causing an intentional watchdog reset. Full system reboot → `pyd_init()` → PIR recovery. This is NOT the normal PIR recovery path — it only triggers during UART/AP communication failures.

**Watchdog reset detection at boot** (user.c:1052-1066):
```c
if (!(nrf_power_resetreas_get() & POWER_RESETREAS_DOG_Msk)) {
    monet_data.cool_boot_flag = 0x80;  // Not a watchdog reset
}
if (resetfromDFU || (nrf_power_resetreas_get() & POWER_RESETREAS_DOG_Msk)) {
    monet_data.tempFromNvram = 1;
}
```
Watchdog resets are detected and logged but NOT treated specially for PIR — the standard `pyd_init()` path is used.

---

### 2.6 BLE-Driven Recovery: NONE

**Every BLE event handler was searched — zero PIR/GPIOTE re-init calls found:**

| Handler | File:Line | PIR Calls |
|---------|-----------|-----------|
| `ble_evt_handler()` | ble_user.c:1030 | None |
| `on_ble_peripheral_evt()` | ble_aus.c:67 / ble_user.c:487 | None |
| `on_ble_central_evt()` | ble_user.c:585 | None |
| `ble_aus.c:on_ble_peripheral_evt()` BLE_GAP_EVT_CONNECTED | ble_aus.c:229 | None |
| `ble_aus.c:on_ble_peripheral_evt()` BLE_GAP_EVT_DISCONNECTED | ble_aus.c:* | None |
| Advertising event handlers | ble_user.c:1239 | None |

**Indirect BLE effect on PIR:** The only BLE/PIR interaction is the sleep guard in `idle_state_handle()` (main.c:355-364):
```c
if ((monet_data.phonePowerOn == 0) || (monet_data.SleepState == SLEEP_NORMAL)) {
    nrf_pwr_mgmt_run();  // sleep
}
```
When the AP is powered on, the main loop does NOT sleep, so `check_pyd_interrupt()` runs every iteration (~10ms). This keeps the 6-hour timer accurate and PIR event processing fast — but it's not a recovery mechanism.

---

## 3. The `pyd_restart()` Recovery Sequence — Detailed Trace

**File:** `camera_pyd1598.c:272-296`

```c
void pyd_restart(void) {
    uint32_t pyd_reg = 0;

    pyd_power_off();                    // [1] PIR_POWER_SW = 1
    pyd_set_status(0);                  // [2] Clear interrupt status
    pir_check_stop();                   // [3] Stop timer
    pyd_gpio_in_disable();              // [4] Unregister from GPIOTE

    nrf_delay_ms(10);                   // [5] Power-off settle (10ms)

    pyd_power_init();                   // [6] Re-init power pin
    nrf_delay_ms(10);                   // [7] Power stabilization (10ms)

    pyd_reg = pyd_params_set(&pyd_params);
    pyd_write_reg(pyd_reg);             // [8] Write 25-bit config (~3ms)

    pyd_gpio_out_low();                 // [9] Output low
    pyd_gpio_in_enable();               // [10] Re-register GPIOTE
}
```

### 3.1 Step-by-step with Timing

| Step | Operation | Duration | GPIOTE State |
|------|-----------|----------|--------------|
| 1 | Power off sensor | ~1µs | Registered |
| 2 | Clear `pyd_interrupt_status` | ~1µs | Registered |
| 3 | Stop `pir_check` timer | ~100µs | Registered |
| 4 | `pyd_gpio_in_disable()` — unregister | ~200µs | **UNREGISTERED** ← DEAD WINDOW BEGINS |
| 5 | `nrf_delay_ms(10)` | 10ms | **UNREGISTERED** |
| 6 | `pyd_power_init()` — configure power pin | ~5µs | **UNREGISTERED** |
| 7 | `nrf_delay_ms(10)` | 10ms | **UNREGISTERED** |
| 8 | `pyd_write_reg()` — 25-bit serial | ~3ms | **UNREGISTERED** |
| 9 | `pyd_gpio_out_low()` — non-GPIOTE output | ~1µs | **UNREGISTERED** |
| 10 | `pyd_gpio_in_enable()` — re-register | ~2ms | **RE-REGISTERED** ← DEAD WINDOW ENDS |

**Total dead window: ~23ms** (steps 4-10). During this time, any PIR edge is silently lost with no detection.

### 3.2 What Is NOT Done

- **No explicit LATCH clearing:** No `NRF_GPIOTE->EVENTS_PORT = 0` or equivalent
- **No PORT event acknowledgment:** The PORT event from the old registration may still be pending
- **No SENSE transition through NOSENSE:** `pyd_gpio_out_low()` uses raw `nrf_gpio_cfg_output()` which doesn't go through GPIOTE, so Errata 75 (no NOSENSE before SENSE restore) does NOT apply to this path — the pin goes output → input via raw GPIO, not GPIOTE
- **No error checking on re-registration:** `APP_ERROR_CHECK(err_code)` at line 206 will fault if `nrf_drv_gpiote_in_init()` fails, causing a hard fault rather than graceful recovery

### 3.3 Window Vulnerability

The ~23ms dead window during `pyd_restart()` applies even when the PIR is otherwise working perfectly. A PIR edge arriving during this 23ms window triggers:
- No interrupt → `pyd_get_status()` stays 0
- No `pirDetectedTimestamp` update
- The next `check_pyd_interrupt()` call exits cleanly (no event, timeout already satisfied)

This is inherent to the power-cycle approach — you cannot detect edges while the sensor is powered off.

---

## 4. Conditional vs Unconditional Re-init

### 4.1 Application Layer (`pyd_gpio_in_enable()`): UNCONDITIONAL

```c
// camera_pyd1598.c:198-209
void pyd_gpio_in_enable(void) {
    uint32_t err_code;
    nrf_drv_gpiote_in_config_t config = GPIOTE_CONFIG_IN_SENSE_TOGGLE(false);
    config.pull = NRF_GPIO_PIN_NOPULL;
    err_code = nrf_drv_gpiote_in_init(PIR_OUT, &config, gpiote_event_handler);
    APP_ERROR_CHECK(err_code);                        // ← FAULTS on failure
    nrf_drv_gpiote_in_event_enable(PIR_OUT, true);
}
```

No guard check. The code **blindly assumes** `nrf_drv_gpiote_in_init()` will succeed. If it returns `NRFX_ERROR_INVALID_STATE` (pin already registered), `APP_ERROR_CHECK` triggers a hard fault.

### 4.2 Driver Layer (`nrfx_gpiote_in_init()`): CONDITIONAL

```c
// nrfx_gpiote.c:515-563
nrfx_err_t nrfx_gpiote_in_init(nrfx_gpiote_pin_t pin, ...) {
    if (pin_in_use_by_gpiote(pin)) {
        err_code = NRFX_ERROR_INVALID_STATE;  // ALREADY REGISTERED
    } else {
        channel = channel_port_alloc(pin, ...);
        if (channel != NO_CHANNELS) {
            // configure pin, set SENSE, register handler
        } else {
            err_code = NRFX_ERROR_NO_MEM;      // NO FREE SLOTS
        }
    }
    return err_code;
}
```

The driver **does** check for duplicate registration. If `PIR_OUT` is already registered, the call fails with `NRFX_ERROR_INVALID_STATE`.

### 4.3 Implication

In normal operation, `pyd_gpio_in_enable()` is always called **after** `pyd_gpio_in_disable()`, so the pin is freed and re-registration succeeds. The driver-level guard is never needed in the happy path.

However, this means `pyd_gpio_in_enable()` **cannot be used as a "refresh"** — if PIR_OUT is somehow still registered but non-functional (e.g., GPIOTE channel corruption, SoftDevice interference), calling `pyd_gpio_in_enable()` without first calling `pyd_gpio_in_disable()` would **hard fault** the system via `APP_ERROR_CHECK`.

This is critical: it means the developers **must always pair disable-then-enable**. They cannot simply "re-arm" the PIR interrupt. Any code path that tries to call `pyd_gpio_in_enable()` without first disabling will crash.

---

## 5. Is the Recovery Path a Workaround?

**YES. Strong evidence the developers were working around a known issue:**

### 5.1 Evidence

1. **Blind timer, no health check:** `pyd_restart()` fires based on time-since-last-event, not on any diagnostic indicating the PIR is actually broken. This is a classic "if it's been quiet too long, assume it's broken" pattern.

2. **Unconditional re-init with fault-on-failure:** `pyd_gpio_in_enable()` has no error recovery — it faults the system if GPIOTE re-init fails. This suggests the developers were confident the re-init path "always works" and didn't plan for failure handling.

3. **Log message with decorative separators:** The `"++++++++++++++++++++++++++++pyd_restart++++++++++++++++++++++++++++"` log line (user.c:789) uses a distinctive formatting pattern reserved for significant debug events. The `pyd_set_first_interrupt()` path (line 727) uses identical formatting, suggesting both were added during the same debugging session — likely when the developers were investigating PIR reliability issues.

4. **No LATCH clearing:** The recovery path doesn't clear PORT events or LATCH state, which would be the natural first step if the developers understood the hardware-level cause of dropped events. Instead, they power-cycle the entire sensor — a brute-force approach.

5. **`pyd_restart()` is also used for threshold changes** (camera_pyd1598.c:65): The same full-restart function is reused for configuration changes, suggesting it was designed as a general-purpose "reset everything" primitive rather than a targeted recovery path.

6. **6-hour period is arbitrary:** No hardware constraint, timer limitation, or power budget explains the 6-hour choice. It's long enough to avoid battery drain from frequent power-cycles (~23ms active time per 6 hours = 0.0001% duty cycle — negligible), short enough that missing PIR detection for 6 hours is "acceptable" for a trail camera.

### 5.2 What This Implies About the Root Cause

The fact that a 6-hour restart "fixes" the symptom constrains the root cause:

| Root Cause Class | Compatible with 6-hour recovery? |
|------------------|----------------------------------|
| GPIOTE registration silently lost (corrupted state) | **YES** — restart re-registers |
| SENSE configuration corrupted | **YES** — restart reconfigures |
| PIR sensor locked up / in bad state | **YES** — power-cycle clears it |
| PIR events dropped but state still valid | **PARTIALLY** — restart doesn't fix the dropping, but it does create a "fresh start" window |
| Slot exhaustion (Track 1: ruled out for GA01/GA02) | **NO** — restart wouldn't free slots taken by other pins |
| Permanent hardware fault | **NO** — restart can't fix broken hardware |

The most compatible root cause is a **gradual accumulation of dropped events** reaching a point where `pyd_interrupt_status` is stale or the sensor state is unreliable, combined with the periodic restart providing a clean slate.

---

## 6. Interaction with Other Tracks

### 6.1 Track 5 (Handler Drop) → Track 6

Track 5 found PIR events are dropped at three layers. The 6-hour restart is the **only mechanism** that can break the cycle if all three layers conspire to silence PIR:

1. If TOGGLE polarity + narrow pulse → Errata 55 at nrfx ISR layer → handler never called
2. If handler IS called but pin already LOW → application layer silently returns
3. After `pyd_gpio_reconfig()` → SENSE dead zone for next event

Under worst-case conditions where all three drop layers are active, the system goes 6 hours without any successful PIR events. Then `pyd_restart()` power-cycles the sensor, re-registers GPIOTE, and provides a fresh start. The sensor then produces events until the cycle repeats.

### 6.2 Track 3 (ISR→Timer Re-entrancy) → Track 6

`check_pyd_interrupt()` is called from TWO contexts:
1. **Main loop** (main.c:641): Direct call, `pir_checking` spin-waits for timer context
2. **Timer callback** (user.c:622-626): `pir_check_handler()` → `check_pyd_interrupt()`

The `pir_checking` spin-wait at main.c:637-639 prevents concurrent execution:
```c
if (pir_is_checking()) {
    nrf_delay_us(1);
    while (pir_is_checking()) nrf_delay_us(1);
}
```

The recovery path (`else if timeout → pyd_restart()`) executes INSIDE `check_pyd_interrupt()` with `pir_checking = true` set. The `pyd_restart()` calls `pir_check_stop()` (line 280) which stops the `pir_check` timer, preventing timer context from firing during restart. This is safe, but the ~23ms blocking delay inside the main loop is significant.

### 6.3 Track 4 (Sleep/Wake) → Track 6

The `count1sec` variable used by the recovery timer is driven by `atel_timerTickHandler()`, which runs on RTC1 via the app_timer library. RTC1 is independent of sleep state — it continues counting during System ON sleep (WFE). The timer is accurate regardless of sleep duration, ensuring the 6-hour recovery fires on time even during extended low-power operation.

### 6.4 Track 7 (SoftDevice BLE Timeslot Interference) → Track 6

Track 7 investigates SoftDevice (s132) BLE radio timeslot interference with GPIOTE ISR servicing. The SoftDevice runs at higher NVIC priority and blocks application interrupt handlers during radio timeslots. This section evaluates how SoftDevice timeslot activity interacts with Track 6's 6-hour recovery mechanism.

**RTC allocation — no conflict:** The 6-hour `count1sec` recovery timer is driven by `app_timer` on RTC1 (sdk_config.h: `APP_TIMER_CONFIG_RTC_FREQUENCY 1` = 16384 Hz). The SoftDevice s132 uses RTC0 for its internal timeslot scheduler. The sdk_config.h confirms both `RTC0_ENABLED` and `RTC1_ENABLED` are `0` at the nrf_drv level (lines 5542, 5549), which is expected — SoftDevice and app_timer manage their respective RTCs directly without the nrf_drv_rtc abstraction. These are separate hardware peripherals: **no RTC resource conflict exists** between SoftDevice BLE timing and the 6-hour recovery timer.

| Resource | Owner | Hardware | sdk_config.h Evidence |
|----------|-------|----------|----------------------|
| RTC0 | SoftDevice s132 (BLE timeslots) | Peripheral instance 0 | `RTC0_ENABLED 0`, `NRFX_RTC0_ENABLED 0` (lines 5542, 3422) |
| RTC1 | app_timer (`count1sec`, `pir_check` timer) | Peripheral instance 1 | `RTC1_ENABLED 0`, `NRFX_RTC1_ENABLED 0` (lines 5549, 3429) |
| TIMER1 | Not used | Peripheral instance 1 | `TIMER1_ENABLED 0`, `NRFX_TIMER1_ENABLED 0` (lines 5877, 4100) |

**Main-loop preemption — delay only, not denial:** `check_pyd_interrupt()` (user.c:641) runs from the main loop at application priority, NOT from an ISR. During BLE radio timeslots (SoftDevice priority 0-2), the main loop is preempted. However, this only delays the recovery check by the duration of a single radio event (~1-2ms for a BLE connection event on nRF52832). The 6-hour recovery period (`PIR_RESTART_TIMEOUT = 21600` seconds) is six orders of magnitude larger than the preemption window, so BLE timeslot preemption does **not** materially affect recovery timing.

**Recovery as ultimate fallback for Track 7 risks:** If SoftDevice timeslots contribute to PIR event loss — multiple edges arriving within a single timeslot that exceed LATCH capacity because the GPIOTE ISR cannot service them — the resulting dropped events cascade as described in Track 5. The 6-hour `pyd_restart()` is the **only mechanism** that ensures eventual recovery from timeslot-induced event loss. This makes Track 6 the safety net for Track 7's identified risks: even if SoftDevice timeslots cause a complete PIR event drought, the blind 6-hour restart will power-cycle the sensor and restore capability.

**NVIC priority note:** The GPIOTE IRQ is at application priority 6 (per Track 3 findings, sdk_config.h: `GPIOTE_CONFIG_IRQ_PRIORITY 6`). The SoftDevice s132 reserves NVIC priorities 0, 2, and (optionally) 1 for its internal operations. Since priority 6 is numerically higher than SoftDevice priorities, GPIOTE ISR execution **is** blocked during BLE radio timeslots. However, the 6-hour recovery timer check in `check_pyd_interrupt()` is NOT ISR-driven — it executes from the main loop at application (thread) priority. The `app_timer` callback that advances `count1sec` runs from the RTC1 ISR (priority 15 by default for app_timer, configurable via `APP_TIMER_CONFIG_IRQ_PRIORITY`), but this only increments the counter — the actual `pyd_restart()` decision and execution happen in the main loop, which resumes after the SoftDevice releases the CPU.

**Net assessment:** SoftDevice BLE activity does not interfere with the 6-hour recovery mechanism. The RTC hardware is independent, the recovery check is main-loop-polled at coarse granularity, and the only consequence of timeslot preemption is a harmless sub-millisecond delay in the recovery check. The risk is asymmetric: SoftDevice timeslots CAN prevent GPIOTE IRQ servicing (potentially contributing to event loss per Track 7), but the 6-hour recovery path itself is immune to SoftDevice interference.

---

## 7. Source Verification

| Finding | Evidence | File:Line |
|---------|----------|-----------|
| 6-hour recovery timer | `PIR_RESTART_TIMEOUT (3600*6)` | user.h:98 |
| Timer check | `(count1sec - pirDetectedTimestamp) >= PIR_RESTART_TIMEOUT` | user.c:785 |
| `pyd_restart()` call | `pyd_restart()` with log message | user.c:788-789 |
| `pirDetectedTimestamp` reset on event | `pirDetectedTimestamp = count1sec` | user.c:713 |
| `pirDetectedTimestamp` reset on restart | `pirDetectedTimestamp = count1sec` | user.c:787 |
| `count1sec` increment | `count1sec++` in 1s tick handler | user.c:1378 |
| `pyd_restart()` full sequence | Power-off → 10ms → init → 10ms → write → output → enable | camera_pyd1598.c:272-296 |
| `pyd_gpio_in_enable()` unconditional | No guard, `APP_ERROR_CHECK(err_code)` | camera_pyd1598.c:198-209 |
| `nrfx_gpiote_in_init()` conditional | `if (pin_in_use_by_gpiote) return ERROR` | nrfx_gpiote.c:523-526 |
| Watchdog init | `NRF_DRV_WDT_DEAFULT_CONFIG`, empty handler | platform_hal_drv.c:1076-1087 |
| Watchdog config | `RELOAD_VALUE=20000, BEHAVIOUR=1, IRQ_PRIO=6` | sdk_config.h:4830-4871 |
| UART alive watchdog trigger | `monet_gpio.WDtimer = 0` after 10 failures | user.c:824 |
| BLE handlers — no PIR calls | Search of all `ble_evt_handler`, `on_ble_*` paths | ble_user.c, ble_aus.c |
| `pyd_restart()` also via threshold | `pyd_set_threshold()` → `pyd_restart()` | camera_pyd1598.c:65 |
| `pir_check_start()` guard conditions | `SleepState, SleepStateChange, systick_remains` | user.c:638-647 |

---

## 8. Recommendations

1. **Reduce the recovery window from 6 hours to a diagnostic period.** The 6-hour timeout masks the underlying issue. Reducing to 60 seconds in a debug build would reveal whether events are being dropped within seconds vs. hours.

2. **Add health diagnostics to `check_pyd_interrupt()`.** Before calling `pyd_restart()`, verify GPIOTE registration is intact (check `pin_in_use_by_gpiote`), verify SENSE configuration matches expected TOGGLE value, and log the state. This would distinguish "sensor locked up" from "GPIOTE registration lost" from "events being dropped."

3. **Add a dropped-event counter.** In `gpiote_event_handler()` (camera_pyd1598.c:167), increment a counter in the `else` branch (when pin reads LOW). This is currently a silent return — it should at minimum be logged. Combined with a periodic diagnostic dump, this would reveal the frequency of handler-level drops.

4. **Replace `pyd_restart()` with `pyd_gpio_reconfig()` for the periodic path.** If the issue is GPIOTE registration loss (not sensor lockup), a ~620µs reconfig is far preferable to a ~23ms power-cycle. The power-cycle approach should be reserved for confirmed sensor lockup detection.

5. **Add NOSENSE transition to `pyd_restart()` output-to-input path.** Even though `pyd_gpio_out_low()` bypasses GPIOTE, the subsequent `pyd_gpio_in_enable()` re-applies SENSE after the pin was in output mode. Adding `nrf_gpio_cfg_sense_set(PIR_OUT, NRF_GPIO_PIN_NOSENSE)` before calling `pyd_gpio_in_enable()` would satisfy Errata 75.

---

## 9. Confidence Assessment

| Aspect | Confidence | Rationale |
|--------|-----------|-----------|
| Recovery path exists | **HIGH (10/10)** | Source-verified at every step |
| 6-hour period | **HIGH (10/10)** | `PIR_RESTART_TIMEOUT` constant confirmed |
| No active health check | **HIGH (10/10)** | `check_pyd_interrupt()` only checks timeout vs `pyd_get_status()` |
| No BLE-driven recovery | **HIGH (10/10)** | Full search of all BLE event handlers |
| Recovery is a workaround | **HIGH (9/10)** | Decorative log formatting, no health check, brute-force approach, 6-hour arbitrary period — multiple independent signals |
| Dead window during restart | **HIGH (10/10)** | Timing calculated from source delays |
| No LATCH clearing | **HIGH (10/10)** | Full restart path verified, no PORT event access |
