mod adopt;
mod cli;
mod cwd_guard;
mod launch;
mod release;
mod selfupdate;
mod services;
mod slots;
mod tree;

use anyhow::Context;
use cli::Command;

fn main() -> anyhow::Result<()> {
    let cli = cli::parse();

    match cli.command {
        Some(Command::Launch { args }) => launch(args),
        Some(Command::Install { source, channel }) => install(source, channel),
        Some(Command::Apply {
            source,
            version,
            notify_file,
            relaunch_app,
            report,
        }) => apply(source, version, notify_file, relaunch_app, report),
        Some(Command::Rollback) => rollback(),
        Some(Command::Status { check, json }) => status(check, json),
        Some(Command::Adopt {
            from_checkout,
            source,
            undo,
        }) => adopt(from_checkout, source, undo),
        Some(Command::SelfRestage) => self_restage(),
        None => {
            // Should not happen — parse() fills in a default.
            unreachable!("cli::parse() should always set a command")
        }
    }
}

fn launch(args: Vec<String>) -> anyhow::Result<()> {
    launch::launch(args)
}

fn install(_source: Option<String>, _channel: String) -> anyhow::Result<()> {
    todo!("install: download → verify → stage → preflight → flip (task 1.4)")
}

fn apply(
    _source: Option<String>,
    _version: Option<String>,
    _notify_file: Option<String>,
    _relaunch_app: Option<String>,
    _report: String,
) -> anyhow::Result<()> {
    // The full apply pipeline: download → verify → stage → preflight → flip
    // → restage → restart services. Individual pieces are in slots.rs,
    // release.rs, selfupdate.rs, services.rs. The orchestration is wired
    // when the apply verb is fully implemented.
    //
    // Post-flip ledger application (task 5.2): after the flip commits and
    // before restarting services, run the ledger against the NEW slot:
    //   <new slot>/bin/hermes features apply-ledger --json
    // Failures are warnings (never fail the flip for a feature install).
    todo!("apply: download → verify → stage → preflight → flip → restage (task 1.4)")
}

fn rollback() -> anyhow::Result<()> {
    let hermes_home = dirs::home_dir()
        .context("cannot find home directory")?
        .join(".hermes");
    let version = slots::rollback(&hermes_home)?;
    println!("Rolled back to {}", version);
    Ok(())
}

fn status(check: bool, json: bool) -> anyhow::Result<()> {
    let hermes_home = dirs::home_dir()
        .context("cannot find home directory")?
        .join(".hermes");
    let current = slots::resolve_current(&hermes_home).unwrap_or(None);
    let previous = slots::resolve_previous(&hermes_home).unwrap_or(None);

    if json {
        let status = serde_json::json!({
            "current": current,
            "previous": previous,
            "check": check,
        });
        println!("{}", serde_json::to_string_pretty(&status).unwrap());
    } else {
        println!("hermes-updater 0.1.0");
        match current {
            Some(v) => println!("  current:  {}", v),
            None => println!("  current:  (none)"),
        }
        match previous {
            Some(v) => println!("  previous: {}", v),
            None => println!("  previous: (none)"),
        }
    }
    Ok(())
}

fn adopt(from_checkout: Option<String>, source: Option<String>, undo: bool) -> anyhow::Result<()> {
    let hermes_home = dirs::home_dir()
        .context("cannot find home directory")?
        .join(".hermes");

    let checkout = match from_checkout {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            // Default to the current checkout (PROJECT_ROOT)
            std::path::PathBuf::from(".")
        }
    };

    adopt::adopt(&hermes_home, &checkout, source.as_deref(), undo)
}

fn self_restage() -> anyhow::Result<()> {
    todo!("self-restage: wire to selfupdate::self_restage (task 1.6 impl done, wiring in task 1.4's apply flow)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_works() {
        // status is the one verb that isn't a stub — it prints a version line.
        assert!(status(false, false).is_ok());
    }
}
