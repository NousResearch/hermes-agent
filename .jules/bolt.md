## Performance Optimization: Gateway Startup Event Loop Block

**Optimization:** Replaced blocking `time.sleep()` with `await asyncio.sleep()` in the async `start_gateway` process termination wait loop.
**Impact:** Restores asynchronous execution of background tasks (like background cron timers, background connections, or tick mechanisms) during the gateway startup phase (which could take up to 10 seconds).
**Context:** When a service restarts with `--replace`, it polls the system using `_pid_exists` checking if the previous instance has shut down gracefully before forceful termination. Doing `time.sleep(0.5)` halted the whole event loop. Async sleep allows background event loop tasks to continue their tick cycles and execution.
