import { tryReadSharedJson } from "./shared-assets.js";
const SHARED_ERRORS = tryReadSharedJson("messages/errors.json") ??
    {
        mcp_ops_mutually_exclusive: "ops and ops_file are mutually exclusive — provide one.",
        mcp_ops_required: "Provide either ops or ops_file.",
        cli_ops_mutually_exclusive: "--ops and --ops-file are mutually exclusive.",
        cli_ops_required: "Provide --ops or --ops-file.",
        cli_dry_run_with_attachment: "--dry-run cannot be combined with --attachment.",
    };
export function sharedError(key) {
    return SHARED_ERRORS[key];
}
export function sharedErrorTemplate(key, values) {
    let out = SHARED_ERRORS[key];
    for (const [k, v] of Object.entries(values)) {
        out = out.replaceAll(`{${k}}`, String(v));
    }
    return out;
}
