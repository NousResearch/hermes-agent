use clap::{Parser, Subcommand};

/// Hermes launcher + updater.
///
/// When invoked as `hermes` (default), the `launch` verb runs.
/// When invoked as `hermes-updater` (argv[0] sniff), updater verbs
/// are the default namespace.
#[derive(Parser, Debug)]
#[command(name = "hermes", version, about, propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Args passthrough for `launch` (when no subcommand is given).
    /// These are forwarded to `hermes_cli.main` as-is.
    #[command(flatten)]
    pub launch_args: LaunchArgs,
}

/// Passthrough args for the default `launch` verb.
#[derive(clap::Args, Debug, Default)]
pub struct LaunchArgs {
    /// Remaining args forwarded to the Python CLI.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub args: Vec<String>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Launch the Hermes agent (default when no subcommand is given).
    Launch {
        /// Args to pass to `hermes_cli.main`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    /// Install Hermes from a release bundle (first install).
    Install {
        /// Release source URL (https:// or file://).
        #[arg(long)]
        source: Option<String>,

        /// Release channel (stable or nightly).
        #[arg(long, default_value = "stable")]
        channel: String,
    },

    /// Apply an update: download, verify, stage, preflight, flip.
    Apply {
        /// Release source URL (https:// or file://).
        #[arg(long)]
        source: Option<String>,

        /// Specific version to apply (defaults to latest).
        #[arg(long)]
        version: Option<String>,

        /// Notify a file when done (gateway IPC).
        #[arg(long)]
        notify_file: Option<String>,

        /// Relaunch an app after applying (desktop IPC).
        #[arg(long)]
        relaunch_app: Option<String>,

        /// Report progress as JSON events.
        #[arg(long, default_value = "text")]
        report: String,
    },

    /// Rollback to the previous version.
    Rollback,

    /// Show current/previous versions, staged leftovers, channel.
    Status {
        /// Check for available updates.
        #[arg(long)]
        check: bool,

        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Migrate a legacy git-checkout install to managed slots.
    Adopt {
        /// Path to the legacy checkout to adopt from.
        #[arg(long)]
        from_checkout: Option<String>,

        /// Release source URL.
        #[arg(long)]
        source: Option<String>,

        /// Undo a previous adoption (re-point symlink at old target).
        #[arg(long)]
        undo: bool,
    },

    /// Restage the updater binary from the current slot.
    #[command(name = "self-restage")]
    SelfRestage,
}

/// Detect whether we were invoked as `hermes-updater` (busybox-style).
/// When true, updater verbs are the default; when false, `launch` is.
pub fn invoked_as_updater() -> bool {
    let argv0 = std::env::args().next().unwrap_or_default();
    let basename = argv0.rsplit(['/', '\\']).next().unwrap_or(&argv0);
    // Strip .exe on Windows
    let basename = basename.trim_end_matches(".exe");
    basename == "hermes-updater"
}

/// Parse CLI args. When invoked as `hermes-updater` with no subcommand,
/// default to `status` (the most common updater query). When invoked as
/// `hermes` with no subcommand, default to `launch`.
pub fn parse() -> Cli {
    let mut cli = Cli::parse();

    // If no subcommand was given, pick a default based on argv[0].
    if cli.command.is_none() {
        if invoked_as_updater() {
            // `hermes-updater` with no args → show status
            cli.command = Some(Command::Status {
                check: false,
                json: false,
            });
        } else {
            // `hermes` with no subcommand → launch
            cli.command = Some(Command::Launch {
                args: std::mem::take(&mut cli.launch_args.args),
            });
        }
    }

    cli
}
