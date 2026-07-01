// Variable substitution for JMAP presets / inline ops ($VAR_NAME tokens).
/** Keys allowed in MCP `vars` / skill `--vars` (without leading `$`). */
export const USER_VAR_KEY_RE = /^[A-Z][A-Z0-9_]*$/;
/**
 * Parses a JSON object of string values (skill `--vars` / MCP `vars`).
 * Throws `Error` with the same messages the CLI used to emit via `fail(...)`.
 */
export function parseUserVarsJson(jsonString) {
    let obj;
    try {
        obj = JSON.parse(jsonString);
    }
    catch (err) {
        throw new Error(`--vars is not valid JSON: ${err.message}`);
    }
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) {
        throw new Error("--vars must be a JSON object of { VAR_NAME: string }.");
    }
    for (const [k, v] of Object.entries(obj)) {
        if (!USER_VAR_KEY_RE.test(k)) {
            throw new Error(`--vars key '${k}' must match /^[A-Z][A-Z0-9_]*$/.`);
        }
        if (typeof v !== "string") {
            throw new Error(`--vars value for '${k}' must be a string.`);
        }
    }
    return obj;
}
/** Matches `$FOO_BAR`; excludes JMAP keywords like `$draft` (lowercase). */
export const VAR_PATTERN = /\$([A-Z][A-Z0-9_]*)/g;
function varPattern() {
    return new RegExp(VAR_PATTERN.source, VAR_PATTERN.flags);
}
/** Names substituted from JMAP session / credentials when not overridden in `vars`. */
export const SESSION_VAR_NAMES = new Set([
    "ACCOUNT_ID",
    "INBOX",
    "INBOX_MAILBOX_ID",
]);
/** Unique variable names in order of first occurrence (without leading `$`). */
export function findVarReferences(raw) {
    const seen = new Set();
    const order = [];
    for (const m of raw.matchAll(varPattern())) {
        const name = m[1];
        if (!seen.has(name)) {
            seen.add(name);
            order.push(name);
        }
    }
    return order;
}
function formatMissingError(missing) {
    const tokens = missing.map((n) => `$${n}`);
    const hasSession = missing.some((n) => SESSION_VAR_NAMES.has(n));
    let msg = `Missing values for variables: ${tokens.join(", ")}. ` +
        "Pass custom placeholders in vars (MCP) or --vars (skill).";
    if (hasSession) {
        msg +=
            " For $ACCOUNT_ID, $INBOX, and $INBOX_MAILBOX_ID, ensure register completed " +
                "and credentials are valid, or pass overrides in vars.";
    }
    return new Error(msg);
}
/**
 * Replaces every `$VAR_NAME` in `raw` with the corresponding string.
 * Single pass — values are not scanned for further `$` tokens.
 * Throws if any referenced variable has no value (after vars + autoResolvers).
 */
export async function substituteVars(input) {
    const names = findVarReferences(input.raw);
    if (names.length === 0) {
        return { text: input.raw };
    }
    const userVars = input.vars ?? {};
    const resolved = new Map();
    for (const name of names) {
        if (Object.prototype.hasOwnProperty.call(userVars, name)) {
            resolved.set(name, userVars[name]);
            continue;
        }
        const resolver = input.autoResolvers?.[name];
        if (resolver) {
            resolved.set(name, await resolver());
            continue;
        }
    }
    const missing = names.filter((n) => !resolved.has(n));
    if (missing.length > 0) {
        throw formatMissingError(missing);
    }
    const text = input.raw.replace(varPattern(), (_full, name) => {
        return resolved.get(name);
    });
    return { text };
}
