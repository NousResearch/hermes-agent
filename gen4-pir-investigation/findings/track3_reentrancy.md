# Track 3 — ISR→Timer→check_pyd_interrupt Re-entrancy (Gen4)

**Date:** 2026-05-28
**Codebase:** atel-reveal-4-mcu (branch: pir-analysis-gen4)
**Target:** GA02 (nRF52832 + S132 SoftDevice, bare-metal)
**Goal:** Map GPIOTE ISR → app_timer_start → timer callback flow. Analyze check_pyd_interrupt re-entrancy safety. Determine all NVIC/FreeRTOS priorities. Check app_timer_start behavior on rapid re-trigger. Assess ISR-context code execution impact.

---

## 1. Executive Summary

**Verdict: Re-entrancy is architecturally safe but system-latency-hostile.**

The `pir_checking` volatile flag provides an effective guard against re-entrancy on this bare-metal Cortex-M4 where all application interrupts share priority 6 (no nesting). However, `check_pyd_interrupt()` runs in the RTC1 ISR context (priority 6) for ~3-5ms, during which ALL application interrupts (SAADC, UART, TWIS, SPIM, etc.) are blocked. This is a system-wide latency tax paid on every PIR event.

---

## 2. System Configuration

### 2.1 Platform

| Parameter | Value |
|-----------|-------|
| MCU | nRF52832 (Cortex-M4) |
| OS | **Bare-metal** (no FreeRTOS) |
| SoftDevice | S132 v6.x (priorities 0, 1, 4, 5 reserved) |
| APP_TIMER_CONFIG_RTC_FREQUENCY | 1 → RTC @ 16384 Hz |
| APP_TIMER_TICKS(1ms) | 16 ticks |
| Tick resolution | ~61 µs per tick |

### 2.2 NVIC Priority Map

All application interrupts are configured at **priority 6**. The SoftDevice reserves priorities 0, 1, 4, and 5. Priorities 2, 3, and 7 are unused by configuration.

| Interrupt | Priority | Source | Config Macro |
|-----------|----------|--------|--------------|
| **SoftDevice** | 0, 1, 4, 5 | S132 SD | (reserved) |
| **GPIOTE** | 6 | PIR_OUT (pin 26) toggle | `GPIOTE_CONFIG_IRQ_PRIORITY` |
| **RTC1** | 6 | App timer compare | `APP_TIMER_CONFIG_IRQ_PRIORITY` |
| **SWI0 (EGU0)** | 6 | App timer op-queue | `APP_TIMER_CONFIG_IRQ_PRIORITY` |
| **SAADC** | 6 | ADC sampling | `NRFX_SAADC_CONFIG_IRQ_PRIORITY` |
| **TWIM0** | 6 | I2C (sensor comms) | `NRFX_TWIM_DEFAULT_CONFIG_IRQ_PRIORITY` |
| **SPIM** | 6 | SPI (flash) | `NRFX_SPIM_DEFAULT_CONFIG_IRQ_PRIORITY` |
| **SPIS** | 6 | SPI slave | `NRFX_SPIS_DEFAULT_CONFIG_IRQ_PRIORITY` |
| **UARTE** | 6 | UART (AP comms) | `NRFX_UARTE_DEFAULT_CONFIG_IRQ_PRIORITY` |
| **COMP** | 6 | Comparator | `COMP_CONFIG_IRQ_PRIORITY` |
| **LPCOMP** | 6 | Low-power comparator | `LPCOMP_CONFIG_IRQ_PRIORITY` |
| **Main loop** | Thread | Bare-metal polling | (interruptible) |

### 2.3 Cortex-M4 Same-Priority Behavior

On Cortex-M4 NVIC, interrupts at the **same priority cannot preempt each other**. Instead, they tail-chain: the NVIC holds the next pending interrupt in the tail-chaining register and dispatches it immediately when the current ISR returns, without the full stacking/unstacking overhead. This is critical for the re-entrancy analysis below.

### 2.4 No FreeRTOS

The codebase does not include `FreeRTOSConfig.h`. The `APP_TIMER_TICKS` macro resolves to the `!FREERTOS` branch in `app_timer.h:113`, confirming bare-metal operation. There is no task scheduler, no preemptive multitasking among application threads. The "main loop" in `main.c:636` runs at thread level (base priority, fully interruptible by all priority 6 ISRs).

---

## 3. Complete Flow Trace: GPIOTE ISR → check_pyd_interrupt

### 3.1 Trigger

PYD1598 PIR sensor detects motion → drives PIR_OUT (pin 26) high.
The GPIOTE channel (configured for TOGGLE sense, line 201 of `camera_pyd1598.c`) latches the rising edge.

### 3.2 GPIOTE ISR (priority 6)

**Source:** `camera_pyd1598.c:167`

```c
static void gpiote_event_handler(nrf_drv_gpiote_pin_t pin, nrf_gpiote_polarity_t action)
{
    if(nrf_gpio_pin_read(PIR_OUT))       // Confirm pin is still high
    {
        pyd_set_status(1);               // pyd_interrupt_status = 1
        NRF_LOG_INFO("low to high\n");
        pir_check_start();               // → timer start
    }
}
```

**Key:** `pyd_set_status(1)` sets `pyd_interrupt_status = 1` (file-static, NOT volatile — see §7). The ISR then invokes `pir_check_start()`.

### 3.3 pir_check_start() — Guard and Timer Start

**Source:** `user.c:749`

```c
void pir_check_start(void)
{
    if(monet_data.SleepState != SLEEP_OFF
       && monet_data.SleepStateChange == 0
       && pf_systick_remains() > APP_TIMER_TICKS(TIME_UNIT)  // >160 ticks (~10ms)
       && !pir_checking)                                     // re-entrancy guard
    {
        NRF_LOG_INFO("pir_check_timer start\n");
        pir_checking = true;                                 // SET before timer
        APP_ERROR_CHECK(app_timer_start(m_pir_check_timer, 5, NULL));
    }
    //else wait pf_systick to check in main thread
}
```

**Guard conditions:**
- `SleepState != SLEEP_OFF`: System is in low-power mode (PIR wakeup relevant)
- `SleepStateChange == 0`: No sleep transition in progress
- `pf_systick_remains() > APP_TIMER_TICKS(10ms)`: At least ~10ms until next systick boundary (avoids timer collision)
- `!pir_checking`: **Re-entrancy guard** — prevents starting a new timer while check_pyd_interrupt is executing

**Timer parameters:**
- Timer ID: `m_pir_check_timer` (APP_TIMER_DEF, user.c:63)
- Timer mode: `APP_TIMER_MODE_SINGLE_SHOT` (user.c:741)
- Timeout: **5 ticks** = ~305 µs (5 × 61 µs)
- Context: `NULL`

The 5-tick delay is the minimum allowed (`APP_TIMER_MIN_TIMEOUT_TICKS = 5`, `app_timer.h:91`). This is effectively zero-delay debouncing — it just defers the work from GPIOTE ISR context to the app_timer (RTC1) ISR context.

### 3.4 app_timer_start() — ISR-Safe Op-Queue

**Source:** `app_timer.c:829` (`timer_start_op_schedule`)

```c
static uint32_t timer_start_op_schedule(timer_node_t * p_node,
                                        uint32_t timeout_initial,
                                        uint32_t timeout_periodic,
                                        void * p_context)
{
    uint8_t last_index;
    uint32_t err_code = NRF_SUCCESS;

    CRITICAL_REGION_ENTER();                          // Briefly disable interrupts
    timer_user_op_t * p_user_op = user_op_alloc(&last_index);
    if (p_user_op == NULL)
        err_code = NRF_ERROR_NO_MEM;                  // Queue full
    else {
        p_user_op->op_type = TIMER_USER_OP_TYPE_START;
        p_user_op->p_node  = p_node;
        p_user_op->params.start.ticks_at_start = rtc1_counter_get();
        p_user_op->params.start.ticks_first_interval = timeout_initial;
        p_user_op->params.start.ticks_periodic_interval = timeout_periodic;
        p_user_op->params.start.p_context = p_context;
        user_op_enque(last_index);
    }
    CRITICAL_REGION_EXIT();

    if (err_code == NRF_SUCCESS)
        timer_list_handler_sched();                   // Pend SWI
    return err_code;
}
```

**ISR safety:** The op-queue insertion is protected by `CRITICAL_REGION_ENTER/EXIT` (disables interrupts briefly on the Cortex-M4). Since GPIOTE ISR is priority 6 and SWI/RTC1 are also priority 6, the critical section only protects against SoftDevice interrupts (priorities 0-5), which are the only ones that can preempt.

**Queue sizing:** `APP_TIMER_CONFIG_OP_QUEUE_SIZE = 10` (sdk_config.h:6430). If the queue fills, `app_timer_start` returns `NRF_ERROR_NO_MEM` → `APP_ERROR_CHECK` → system reset.

### 3.5 SWI Handler — Timer List Update

After GPIOTE ISR returns, the pending SWI fires (tail-chained at same priority):

```c
void SWI_IRQHandler(void)
{
    timer_list_handler();                    // app_timer.c:925-928
}
```

`timer_list_handler()` (app_timer.c:725) processes the op-queue:
- Dequeues START operation
- If timer is already running (`p_timer->is_running`): **skips** (line 618-621)
- Otherwise: sets `ticks_to_expire`, marks `is_running = true`, inserts into timer list
- Calls `compare_reg_update()` → sets RTC1 CC0 register, starts RTC1 if not running

### 3.6 RTC1 ISR — Timer Expiry and Callback

~5 ticks (~305 µs) later, RTC1 compare 0 matches:

```c
void RTC1_IRQHandler(void)                   // app_timer.c:906
{
    NRF_RTC1->EVENTS_COMPARE[0] = 0;         // Clear event
    // ... clear all other events ...
    timer_timeouts_check();                  // app_timer.c:917
}
```

`timer_timeouts_check()` (app_timer.c:400):
- Computes elapsed ticks from RTC counter
- Walks the timer list, finds expired timers
- For each expired timer: calls `timeout_handler_exec(p_timer)`

`timeout_handler_exec()` (app_timer.c:383):
- `APP_TIMER_CONFIG_USE_SCHEDULER = 0` → directly calls handler
- `p_timer->p_timeout_handler(p_timer->p_context)` → **`pir_check_handler(NULL)`**

### 3.7 pir_check_handler → check_pyd_interrupt

```c
static void pir_check_handler(void * p_context)   // user.c:733
{
    UNUSED_PARAMETER(p_context);
    check_pyd_interrupt();                         // user.c:736
}
```

### 3.8 check_pyd_interrupt() — The Heavy Function

**Source:** `user.c:815-901` (87 lines, summarized below)

```
check_pyd_interrupt():
    pir_checking = true                           // SET on entry (line 819)
    
    if pyd_get_status():                          // PIR triggered
        pirDetectedTimestamp = count1sec
        pyd_set_status(0)                         // Clear status
        pir_value = pyd_gpio_reconfig()           // BIT-BANG ~3.75ms
        NRF_LOG_RAW_INFO(...)
        NRF_LOG_FLUSH()
        
        if !pyd_check_first_interrupt():          // Debounce first trigger
            pyd_set_first_interrupt(1)
        else:
            monet_xF2command(pir_value)           // Factory mode: send PIR value
            if pir_value == -1:                   // Read error
                pir_count++
                if pir_count >= PIR_TIMEOUT(10s): // 10-second error timeout
                        pir_checking = false; return  // Guard cleared before early return
            else:
                pir_count = 0
            
            if conditions_met:
                mcu_slave_reason_update(PIR)
                mcu_wakeup_ap_pir()
                nrf_delay_ms(1)
                MCU_TurnOn_AP()                   // Power on AP
                monet_xE3command()
                pir_interval_delay = BASELINE_PIR_DELAY + pir_time_interval
                pir_is_valid = 0
                pir_triggered_secs = 0
                monet_gpio.Intstatus |= MASK_FOR_BIT(INT_PIR)
    
    else if PIR_RESTART_TIMEOUT elapsed:          // 6 hours since last PIR
        pyd_restart()
    
    pir_checking = false                          // CLEAR on exit (line 901)
```

**Execution context:** Runs at RTC1 IRQ priority 6. Duration: ~3-5ms (dominated by `pyd_gpio_reconfig()` bit-banging at ~150µs/bit × 25 bits). Contains logging calls and `MCU_TurnOn_AP()`.

**Note:** In the `pir_value == -1` error path at the `pir_count >= PIR_TIMEOUT` threshold, `pir_checking` is correctly set to `false` before the early return (see §5.1). No deadlock.

---

## 4. Re-entrancy Analysis

### 4.1 ISR→Timer Callback (Same Priority, No Nesting)

Since GPIOTE ISR (priority 6) and RTC1 ISR (priority 6) share the same NVIC priority, the Cortex-M4 NVIC **cannot nest them**. The sequence is strictly serial:

```
GPIOTE ISR fires → runs to completion → returns
  → tail-chained SWI fires → runs to completion → returns
    → RTC1 fires ~305µs later → runs to completion → returns
```

There is NO window where `pir_check_start()` and `check_pyd_interrupt()` execute concurrently from ISR context. The `pir_checking` flag guard in `pir_check_start()` is therefore redundant for ISR→ISR protection (though it does protect against main-loop concurrency).

### 4.2 Main Loop Concurrency

`check_pyd_interrupt()` is ALSO called from the main loop (`main.c:665`):

```c
// main.c:661-665 (inside the infinite for(;;) loop at line 636)
if (pir_is_checking()) {
    nrf_delay_us(1);
    while (pir_is_checking()) nrf_delay_us(1);     // Spin-wait guard
}
check_pyd_interrupt();                              // Main-loop invocation
```

**Scenario A: Timer fires while main loop is running**
1. Main loop is at thread level (interruptible)
2. Timer fires at priority 6 → `check_pyd_interrupt()` runs → sets `pir_checking = true`
3. Timer callback returns → `pir_checking = false`
4. Main loop resumes, sees `pir_is_checking() == false`, calls `check_pyd_interrupt()` again
5. `check_pyd_interrupt()` checks `pyd_get_status()` → likely 0 (already cleared by timer callback)
6. **Result:** Handled correctly. Double invocation is harmless (idempotent `pyd_get_status()` check).

**Scenario B: Main loop is in `check_pyd_interrupt()` when GPIOTE fires**
1. Main loop enters `check_pyd_interrupt()` → sets `pir_checking = true` (line 819)
2. GPIOTE ISR fires (priority 6, preempts main loop)
3. ISR calls `pir_check_start()` → `!pir_checking` is FALSE → **bails**
4. BUT `pyd_set_status(1)` WAS called (from `gpiote_event_handler` line 171, BEFORE `pir_check_start`)
5. GPIOTE ISR returns → main loop resumes `check_pyd_interrupt()`
6. Main loop completes, sets `pir_checking = false` (line 901)
7. **Next main loop iteration:** main loop sees `pir_is_checking() == false`, calls `check_pyd_interrupt()` again
8. `check_pyd_interrupt()` checks `pyd_get_status()` → it's 1 → processes the event
9. **Result:** Event is NOT lost — it's deferred by one main loop iteration. Latency bound: 1 main-loop cycle (~10ms in normal sleep, up to 1s in hibernation at `TIME_UNIT_IN_SLEEP_HIBERNATION = 1000ms`).

**Scenario C: Main loop is spinning on `pir_is_checking()` when timer fires**
1. Timer callback fired earlier, `pir_checking = true`, main loop spinning at line 663
2. Timer callback returns, `pir_checking = false`, main loop exits spin
3. Main loop calls `check_pyd_interrupt()` — but PIR state already handled by timer callback
4. `pyd_get_status()` returns 0 → no action
5. **Result:** Handled correctly. The spin-wait ensures serialization.

### 4.3 Rapid Re-trigger: app_timer_start on Already-Running Timer

**Scenario:** PIR sensor fires twice within <305 µs (before the 5-tick timer expires).

1. First GPIOTE → ISR → `pir_check_start()`:
   - `pir_checking = true` → `app_timer_start(m_pir_check_timer, 5, NULL)` → enqueues START op
2. Second GPIOTE → ISR → `pir_check_start()`:
   - `!pir_checking` is FALSE (still true from step 1) → **bails entirely**
   - Timer NOT restarted, timer NOT reset

**`app_timer_start` behavior on already-running timer:**

In `list_insertions_handler()` (app_timer.c:568-669), when the SWI processes the START operation:

```c
case TIMER_USER_OP_TYPE_START:
    break;  // fall through
...
if (p_timer->is_running) {
    continue;  // SKIP — timer already running, start is silently dropped
}
```

Since `pir_check_start()` sets `pir_checking = true` BEFORE calling `app_timer_start`, and `check_pyd_interrupt()` keeps it true until completion, the second GPIOTE event's `pir_check_start()` never even calls `app_timer_start`. But even if it did (e.g., in a hypothetical race), `app_timer_start` on an already-running `APP_TIMER_MODE_SINGLE_SHOT` timer is silently dropped — the timer does NOT restart from the new start time.

**Impact:** Rapidly successive PIR events are merged. The second event is not lost (status is still set), but temporal granularity is reduced. For a PIR sensor with typical retrigger intervals >100ms, this is acceptable. For fast-moving targets that could trigger the sensor at <1ms intervals, the second trigger would be merged with the first.

---

## 5. Bugs and Risks Found

### 5.1 REVIEWED: PIR Read Error Path (pir_value == -1) — No Deadlock

**Location:** `user.c:847-852`

```c
if(pir_value == -1)
{
    if(pir_count >= PIR_TIMEOUT)
    {
        pir_checking = false;           // Guard cleared BEFORE return
        //monet_data.pir_is_valid = 0;
        NRF_LOG_RAW_INFO("pyd error\n");
        return;                         // Safe: pir_checking already false
    }
    else
        pir_count++;
}
```

**Verdict: No deadlock. `pir_checking` is correctly cleared at line 849 before the early return at line 852.**

The re-entrancy guard `pir_checking` is set to `false` as the first statement inside the `pir_count >= PIR_TIMEOUT` branch, *before* the `return`. By the time execution leaves `check_pyd_interrupt()`, the guard is already released. Future GPIOTE ISR invocations of `pir_check_start()` will see `!pir_checking == true` and proceed normally.

**Secondary path:** When `pir_count < PIR_TIMEOUT`, the function does NOT return early — it increments `pir_count` and falls through to the normal exit path, where `pir_checking = false` is set at line 901. This is also safe.

**Residual concern — what if `pir_checking` is corrupted?** If a SoftDevice interrupt (priority 0–5) were to overwrite the `pir_checking` variable to `true` and the write persists, all future PIR detections would be permanently blocked. The only recovery mechanism is the watchdog timeout → `NVIC_SystemReset`. This is covered in the T3→T6 cross-track analysis (§6.4).

### 5.2 pyd_interrupt_status Is NOT volatile

**Location:** `camera_pyd1598.c`

The `pyd_interrupt_status` variable (the backing store for `pyd_get_status()`/`pyd_set_status()`) is a file-static `uint8_t` that is NOT declared `volatile`:

```c
// camera_pyd1598.c (inferred — not shown in our read but confirmed by Track 1 analysis)
static uint8_t pyd_interrupt_status;   // NOT volatile
```

**Access contexts:**
- **Write:** GPIOTE ISR (priority 6) — `pyd_set_status(1)` at line 171, `pyd_set_status(0)` at line 278
- **Read:** Timer callback (priority 6) — `pyd_get_status()` at line 820
- **Read:** Main loop (thread level) — `pyd_get_status()` at line 820 (via main-loop call to `check_pyd_interrupt`)
- **Write:** `pyd_power_off()` at line 278 — called from `check_pyd_interrupt()` → `pyd_gpio_reconfig()` → `pyd_power_off()` (thread or timer context)

**Risk assessment: LOW.** On Cortex-M4, a single-byte (uint8_t) load/store is atomic. The compiler could theoretically optimize away a read of a non-volatile in a tight loop, but `check_pyd_interrupt()` is not a tight loop — it's called once per main-loop iteration with function call barriers (NRF_LOG, nrf_delay, etc.) that prevent the compiler from caching the value across calls. The practical risk is negligible.

**Recommendation:** Declare `volatile` for correctness and to prevent future compiler optimization surprises. Same as the T2 finding for `monet_data`.

### 5.3 System-Wide Latency Tax: All App IRQs Blocked for ~3-5ms

`check_pyd_interrupt()` runs for ~3-5ms at priority 6. During this time, all other application interrupts are blocked:

| Blocked Peripheral | Impact |
|-------------------|--------|
| SAADC (ADC sampling) | Sample FIFO may overflow if sampling rate >200Hz |
| UARTE (AP comms) | Hardware FIFO (6 bytes) may overflow at >19200 baud |
| TWIM0 (I2C sensors) | Clock stretching extended; slave may timeout |
| SPIM (flash) | Flash operations delayed |
| SPIS (slave SPI) | Slave may miss transactions |

**SoftDevice interrupts (priorities 0, 1, 4, 5) are NOT blocked** — they can preempt priority 6. However, SoftDevice timeslot handlers may call application callbacks (at priority 5 or 6 depending on configuration), and those callbacks may access shared state.

---

## 6. Cross-Track Implications

### 6.1 T3→T2 (Re-entrancy → Volatile Race)

The `pir_checking` flag is `volatile bool` (user.c:38), correctly declared. However, `pyd_interrupt_status` (see §5.2) is NOT volatile. Since both variables are accessed from ISR and thread context, the T2 volatile analysis applies: the compiler could theoretically cache `pyd_interrupt_status` across function calls if optimization settings are aggressive enough.

### 6.2 T3→T5 (Re-entrancy → Stack Depth)

`check_pyd_interrupt()` is called from the app_timer callback at priority 6, which runs on the **main stack** (not a process stack — no RTOS). The call chain depth:

```
RTC1_IRQHandler
  → timer_timeouts_check
    → timeout_handler_exec
      → pir_check_handler
        → check_pyd_interrupt
          → pyd_gpio_reconfig         (bit-bang, ~3.75ms)
            → pyd_gpio_in_disable
            → pyd_gpio_read_value
            → pyd_gpio_out_low
            → pyd_gpio_in_enable
          → MCU_TurnOn_AP             (AP power on)
          → monet_xE3command
          → NRF_LOG_RAW_INFO (×5-8)
          → NRF_LOG_FLUSH
```

Each level adds stack frames. With NRF_LOG calls (which can be deep with formatting), this is a non-trivial stack consumption. If a SoftDevice interrupt (priority 0-5) fires during `check_pyd_interrupt()` and uses the main stack, the combined depth could approach the nRF52832's limited stack.

### 6.3 T3→T1 (Re-entrancy → GPIOTE Slot Exhaustion)

`check_pyd_interrupt()` calls `pyd_gpio_reconfig()` which tears down and re-creates the GPIOTE channel for PIR_OUT. If this re-creation fails (all 8 channels exhausted), `APP_ERROR_CHECK` triggers a reset. This is the same mechanism analyzed in T1, now triggered from within the timer callback path.

### 6.4 T3→T6 (Re-entrancy → Fault Tolerance / Recovery)

Track 6 analyzes system-wide fault tolerance and recovery mechanisms (watchdog, reset paths, crash resilience). Track 3 findings have several intersection points:

**6.4.1 IRQ Blocking Window (3-5ms) → Peripheral FIFO Overflow**

`check_pyd_interrupt()` runs at priority 6 for ~3-5ms, blocking all application interrupts. During this window, hardware peripherals continue operating autonomously but their ISRs cannot run to drain FIFOs:

| Peripheral | FIFO Depth | Risk During 3-5ms Window |
|-----------|-----------|--------------------------|
| SAADC | 8 samples (EasyDMA) | Overflow if sampling >2.6 kHz; at typical 200 Hz rates, 1 sample fits in FIFO with margin. Marginal risk in practice. |
| UARTE | 6-byte RX FIFO | Overflow at baud rates >16 kbps. At 115200 bps (~11.5 bytes/ms), the FIFO fills in ~0.5ms — guaranteed overflow during the 3-5ms window. |
| TWIM0 | None (byte-at-a-time) | Clock stretching extends transaction; slave may timeout if SCL held >5ms. |
| SPIS | 1-byte RX (single buffer) | Slave misses any transaction that arrives during the blocked window. |

**Track 6 implication:** If UARTE FIFO overflow causes data loss from the AP communication link, the system has NO application-level recovery — the lost bytes are gone. The AP must implement retry/timeout at the protocol layer. SAADC overflow would produce corrupted ADC samples in the EasyDMA buffer, potentially propagating bad sensor data through the pipeline. Track 6 should document these as hardware-level failure modes without software recovery.

**6.4.2 pir_checking Corruption → Permanent PIR Deadlock**

If `pir_checking` (a `volatile bool` at `user.c:38`) were corrupted to `true` by a SoftDevice ISR (priorities 0-5 can preempt priority 6 and write to RAM), the re-entrancy guard would permanently block all future `pir_check_start()` calls. `pir_is_checking()` would always return `true`, the main-loop spin-wait at `main.c:663` would never exit, and `check_pyd_interrupt()` would never execute again from any context.

**Recovery:** The only mechanism is the watchdog timer → `NVIC_SystemReset`. There is no application-level detection or correction of `pir_checking` corruption. The watchdog timeout is configurable (typically 2-8 seconds on this platform). During the deadlock window between corruption and reset, the system is blind to PIR events.

**Track 6 implication:** This is a single-point-of-failure flag. Track 6 should note that adding a timeout guard (e.g., a timestamp that auto-clears `pir_checking` after N seconds) would provide a defense-in-depth layer against corruption-induced deadlock, avoiding reliance on watchdog alone.

**6.4.3 pyd_gpio_reconfig() GPIOTE Re-acquisition Failure**

`check_pyd_interrupt()` calls `pyd_gpio_reconfig()` which tears down and re-allocates a GPIOTE channel for PIR_OUT on every PIR event. If all 8 channels are exhausted (see T1 analysis), `APP_ERROR_CHECK` triggers `NVIC_SystemReset`. **Recovery path:** Identical to the T1→T6 finding — the only recovery is system reset. No graceful degradation.

**6.4.4 Recovery Path Summary**

All Track 3 failure modes converge on a single recovery mechanism:

```
Failure Mode                    → Detection          → Recovery
─────────────────────────────────────────────────────────────────
pir_checking corruption         → Watchdog timeout   → NVIC_SystemReset
GPIOTE re-acquisition failure   → APP_ERROR_CHECK    → NVIC_SystemReset
app_timer queue exhaustion      → APP_ERROR_CHECK    → NVIC_SystemReset
UARTE/SAADC FIFO overflow       → Silent data loss   → None (AP retry at protocol layer)
```

There is no application-level fault handler, no safe-state fallback, and no partial-degradation mode. Every fatal path terminates in `NVIC_SystemReset`. Track 6 should assess whether the watchdog timeout provides adequate coverage for the worst-case deadlock window (corruption → reset, typically 2-8 seconds) and whether silent data loss (FIFO overflow) is acceptable for the AP communication protocol.

---

## 7. app_timer_start Behavior Summary

| Scenario | Behavior |
|----------|----------|
| Timer not running | Enqueues START op; SWI inserts into list; starts RTC1 |
| Timer already running (SINGLE_SHOT) | Op is silently dropped (`is_running` check at line 618) |
| Timer already running (REPEATED) | Op is silently dropped (same check) |
| Called from ISR context | Safe: CRITICAL_REGION_ENTER protects op-queue |
| Called from main loop | Safe (same mechanism) |
| Queue full (10 entries) | Returns `NRF_ERROR_NO_MEM` → `APP_ERROR_CHECK` → reset |

**Key takeaway:** `app_timer_start` on an already-running `m_pir_check_timer` does NOT restart or extend the timer. The original 5-tick timeout remains. This means rapid PIR events within the 5-tick window are merged, not queued.

---

## 8. Findings Summary

| # | Finding | Severity | Impact |
|---|---------|----------|--------|
| 1 | Re-entrancy is architecturally safe due to same-priority (6) ISRs not nesting on Cortex-M4 | — | No data corruption |
| 2 | `pir_checking` flag guard is effective for ISR→main-loop serialization | — | Works correctly |
| 3 | Rapid re-trigger within 305µs is silently debounced (merged) | LOW | Acceptable for PIR sensor physics |
| 4 | `pyd_interrupt_status` is NOT volatile | LOW | Benign on Cortex-M4 for uint8_t; recommend fix |
| 5 | `check_pyd_interrupt()` blocks all app IRQs for ~3-5ms | **MEDIUM** | SAADC/UART FIFO risk at high data rates |
| 6 | `pyd_gpio_reconfig()` can fail GPIOTE channel alloc → reset | LOW | Same as T1 finding; triggered from timer path |
| 7 | Stack depth in RTC1 IRQ → check_pyd_interrupt is non-trivial | MEDIUM | Combined with SoftDevice nesting, may approach stack limit |

---

## 9. Recommendations

1. **Move `check_pyd_interrupt()` heavy work out of ISR context.** The `pyd_gpio_reconfig()` bit-bang (~3.75ms) should be deferred to the main loop. The timer callback should only set a flag, and the main loop should do the heavy work. This eliminates the system-wide IRQ blocking.

2. **Declare `pyd_interrupt_status` as `volatile`.** Consistency with the rest of the shared-state design.

3. **Consider `APP_TIMER_CONFIG_USE_SCHEDULER = 1`.** This would move timer callbacks to `app_scheduler` (main loop context), which is appropriate for long-running handlers like `check_pyd_interrupt()`.

4. **Add stack watermark monitoring.** Given the call depth from RTC1 ISR through `check_pyd_interrupt()`, adding stack high-water mark monitoring would provide confidence in stack margin.

5. **Increase the timer delay from 5 ticks to a meaningful value.** The current 5-tick (~305µs) delay is essentially zero. A value of `APP_TIMER_TICKS(1)` (~16 ticks, ~1ms) would be a more standard debounce period and would not affect user-visible latency.

---

*End of Track 3 analysis.*
