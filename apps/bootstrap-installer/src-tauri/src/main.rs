// Hermes Setup — process entrypoint. All logic lives in lib.rs so it can
// be unit-tested as a library; this file just calls into it.
//
// The windows_subsystem attribute MUST live here on the binary create
// (not lib.rs) — placing it on the lib was the bug that left a stray
// cmd window behind Hermes-Setup.exe on release builds.
//
// `windows_subsystem = "windows"` strips the console allocation that
// the default `windows_subsystem = "console"` would do, so double-clicking
// the .exe gives you ONLY the Tauri window.
//
// debug_assertions guard: dev builds keep the console so tracing output
// is visible during `cargo tauri dev`.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    hermes_bootstrap_lib::run()
}
