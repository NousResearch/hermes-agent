# Code 1.Rust

``` rust
// P1-1: V30 HARDENING — scoped-allow for architectural process-group termination.
// This unsafe block is NOT a casual shortcut. It is the only Unix mechanism to
// terminate an entire spawned process group when a command times out. The child
// was spawned with process_group(0) — kill() on the parent PID alone would leave
// orphaned children. No safe Rust wrapper exists for killpg(); this is the minimal
// correct primitive. Evidence:
// - tokio::process::Command does not expose PID for safe killpg() dispatch
// - std::process::Command::kill() hits the same syscall internally (always unsafe)
// - Precondition: child_pid is a process group leader spawned in this session
// - Postcondition: entire process group receives SIGKILL, all processes terminated
#[allow(clippy::arc_with_non_send_sync)]
unsafe {
    libc::killpg(child_pid, libc::SIGKILL);
}
```
