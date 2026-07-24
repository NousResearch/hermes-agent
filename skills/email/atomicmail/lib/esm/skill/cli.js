#!/usr/bin/env node
// Atomic Mail AgentSkill — register | jmap_request | help
import "../_dnt.polyfills.js";
import process from "node:process";
import { parseArgs } from "node:util";
import { AgentSession, DEFAULT_API_URL, DEFAULT_AUTH_URL, DEFAULT_JMAP_USING, DEFAULT_POW_SCRYPT_SALT_HEX, defaultFilesFromOutDir, expandCredentialDirInput, getHelp, parseUserVarsJson, persistLoginWithApiKey, readCredentials, readOpsFile, runJmapRequest, sharedError, } from "../lib/mod.js";
const USAGE = `Atomic Mail — AgentSkill

Usage:
  atomicmail <command> [options]

Commands:
  register       PoW signup or login with API key (writes credentials)
  jmap_request   Send a JMAP batch (inline --ops or --ops-file; optional --attachment)
  help           Full documentation [--topic TOPIC] (topic readme = built-in stub)

Examples:
  atomicmail register --username alice
  atomicmail register --api-key UUID
  atomicmail jmap_request --ops-file list_inbox.json
  atomicmail jmap_request --credentials-dir ./.atomic-mail --ops-file send.json --vars '{"TO":"a@b.com","SUBJECT":"Hi"}'
  atomicmail jmap_request --ops-file send_mail_blob_attachment.json --attachment ./notes.txt --vars '{"TO":"self@example.com","SUBJECT":"File","BODY":"See attach."}'
  atomicmail help --topic presets
  atomicmail help --topic readme

Call  atomicmail help  early and often — topics match this install; prefer help
over guessing JMAP or preset details.

Run  atomicmail <command> --help  for command-specific flags.
`;
function exitUsage(code = 0) {
    process.stdout.write(USAGE);
    process.exit(code);
}
function fail(message, code = 1) {
    process.stderr.write(`Error: ${message}\n`);
    process.exit(code);
}
async function cmdRegister(argv) {
    let parsed;
    try {
        parsed = parseArgs({
            args: argv,
            options: {
                "auth-url": { type: "string" },
                "api-url": { type: "string" },
                "scrypt-salt": { type: "string" },
                username: { type: "string" },
                "api-key": { type: "string" },
                "credentials-dir": { type: "string" },
                forced: { type: "boolean" },
                quiet: { type: "boolean" },
                help: { type: "boolean", short: "h" },
            },
            strict: true,
            allowPositionals: false,
        });
    }
    catch (err) {
        fail(err.message, 2);
    }
    if (parsed.values.help) {
        process.stdout.write(`Usage: atomicmail register [OPTIONS]

Register a new inbox (--username) or log in with an existing API key (--api-key).

Options:
  --auth-url URL       Auth-service base URL [env: ATOMIC_MAIL_AUTH_URL, default: https://auth.atomicmail.ai]
  --api-url URL        API / JMAP base URL [env: ATOMIC_MAIL_API_URL, default: https://api.atomicmail.ai]
  --scrypt-salt SALT   PoW salt override [env: ATOMIC_MAIL_SCRYPT_SALT]
  --username NAME      New account (5–21 characters; mutually exclusive with --api-key)
  --api-key KEY        Existing API key (mutually exclusive with --username)
  --credentials-dir DIR  Credential directory (default: ~/.atomicmail)
  --forced             Allow replacing existing credentials with a new account
  --quiet              Less stderr output
  --help, -h           This message

After register, arrange hourly inbox polling per your runtime (atomicmail help
--topic cron): native cron hosts schedule an AGENT turn with list_inbox.json;
no-native-cron hosts ask the operator or remind manual fetch — do not work
around with OS crontab or cross-platform scheduling. Do NOT cron atomicmail
jmap_request alone.
`);
        process.exit(0);
    }
    const env = process.env;
    const authUrl = parsed.values["auth-url"] ??
        env.ATOMIC_MAIL_AUTH_URL ?? DEFAULT_AUTH_URL;
    const apiUrl = parsed.values["api-url"] ??
        env.ATOMIC_MAIL_API_URL ?? DEFAULT_API_URL;
    const scryptSalt = parsed.values["scrypt-salt"] ??
        env.ATOMIC_MAIL_SCRYPT_SALT ?? DEFAULT_POW_SCRYPT_SALT_HEX;
    const dir = parsed.values["credentials-dir"];
    const credentialDir = expandCredentialDirInput(dir);
    const username = parsed.values.username;
    const apiKey = parsed.values["api-key"];
    if (!!username === !!apiKey) {
        fail("Provide exactly one of --username (new account) or --api-key (login).", 2);
    }
    const files = defaultFilesFromOutDir(credentialDir);
    const log = (msg) => {
        if (!parsed.values.quiet)
            process.stderr.write(msg + "\n");
    };
    if (username) {
        log(`Registering "${username}"...`);
        const session = await AgentSession.create({
            authUrl,
            apiUrl,
            scryptSalt,
            credentialDir,
            files,
        });
        const result = await session.register(username, {
            forced: parsed.values.forced === true,
        });
        log(`Wrote credentials under ${credentialDir}`);
        process.stdout.write(JSON.stringify(result, null, 2) + "\n");
        return;
    }
    log("Logging in with API key...");
    const { inboxId } = await persistLoginWithApiKey({
        authUrl,
        apiUrl,
        scryptSalt,
        apiKey: apiKey,
        files,
    });
    log(`Wrote ${files.credentialsFile}`);
    process.stdout.write(JSON.stringify({ inboxId }, null, 2) + "\n");
}
async function cmdJmapRequest(argv) {
    let parsed;
    try {
        parsed = parseArgs({
            args: argv,
            options: {
                "credentials-dir": { type: "string" },
                "credentials-file": { type: "string" },
                "session-file": { type: "string" },
                "capability-file": { type: "string" },
                ops: { type: "string" },
                "ops-file": { type: "string" },
                using: { type: "string" },
                "dry-run": { type: "boolean" },
                vars: { type: "string" },
                attachment: { type: "string", multiple: true },
                "attachment-path-base": { type: "string" },
                help: { type: "boolean", short: "h" },
            },
            strict: true,
            allowPositionals: false,
        });
    }
    catch (err) {
        fail(err.message, 2);
    }
    if (parsed.values.help) {
        process.stdout.write(`Usage: atomicmail jmap_request [OPTIONS]

Send a JMAP request using saved credentials.

Options:
  --credentials-dir DIR      Directory with credentials.json + JWTs (default: ~/.atomicmail)
  --credentials-file PATH    Override credentials.json path
  --session-file PATH        Override session.jwt path
  --capability-file PATH     Override capability.jwt path
  --ops JSON                 Inline JMAP JSON (methodCalls or envelope)
  --ops-file PATH            Preset file ($VAR_NAME placeholders supported)
  --vars JSON                JSON object { VAR_NAME: string } for $VAR_NAME in ops / ops-file
  --attachment PATH          Repeatable; each file is RFC 8620–uploaded before JMAP (injects $ATTACHMENT_N_*)
  --attachment-path-base DIR Base for relative --attachment paths (default: cwd)
  --using LIST               Comma-separated capability URNs (optional)
  --dry-run                  Print resolved request only (not compatible with --attachment)
  --help, -h                 This message
`);
        process.exit(0);
    }
    const dir = parsed.values["credentials-dir"];
    const credentialDir = expandCredentialDirInput(dir);
    const defaults = defaultFilesFromOutDir(credentialDir);
    const credentialsFile = parsed.values["credentials-file"] ??
        defaults.credentialsFile;
    const sessionFile = parsed.values["session-file"] ??
        defaults.sessionFile;
    const capabilityFile = parsed.values["capability-file"] ??
        defaults.capabilityFile;
    const ops = parsed.values.ops;
    const opsFile = parsed.values["ops-file"];
    if (ops && opsFile) {
        fail(sharedError("cli_ops_mutually_exclusive"), 2);
    }
    if (!ops && !opsFile) {
        fail(sharedError("cli_ops_required"), 2);
    }
    const rawAttachments = parsed.values.attachment;
    const attachmentPaths = rawAttachments === undefined
        ? []
        : Array.isArray(rawAttachments)
            ? rawAttachments
            : [rawAttachments];
    if (parsed.values["dry-run"] === true && attachmentPaths.length > 0) {
        fail(sharedError("cli_dry_run_with_attachment"), 2);
    }
    const usingFlag = parsed.values.using;
    const defaultUsing = usingFlag
        ? usingFlag.split(",").map((s) => s.trim()).filter((s) => s.length > 0)
        : [...DEFAULT_JMAP_USING];
    let userVars;
    const varsFlag = parsed.values.vars;
    if (varsFlag !== undefined) {
        try {
            userVars = parseUserVarsJson(varsFlag);
        }
        catch (err) {
            fail(err instanceof Error ? err.message : String(err), 2);
        }
    }
    const creds = await readCredentials(credentialsFile);
    const files = {
        credentialsFile,
        sessionFile,
        capabilityFile,
    };
    const session = await AgentSession.create({
        authUrl: creds.authUrl,
        apiUrl: creds.apiUrl,
        scryptSalt: creds.scryptSalt,
        apiKey: creds.apiKey,
        inboxId: creds.inboxId,
        credentialDir,
        files,
    });
    let raw;
    let sourceLabel;
    if (opsFile) {
        try {
            raw = await readOpsFile(credentialDir, opsFile);
        }
        catch (err) {
            fail(`Could not read --ops-file: ${err.message}`, 2);
        }
        sourceLabel = `ops_file '${opsFile}'`;
    }
    else {
        raw = ops;
        sourceLabel = "ops";
    }
    const attachmentPathBase = parsed.values["attachment-path-base"];
    // Same JMAP path as MCP: built-in `$INBOX` is normalized to a full mailbox
    // address via shared `runJmapRequest` (see `inboxIdToMailboxEmail`).
    const { ok, status, bodyText } = await runJmapRequest({
        session,
        opsJson: raw,
        defaultUsing,
        sourceLabel,
        dryRun: parsed.values["dry-run"] === true,
        vars: userVars,
        attachments: attachmentPaths.length > 0
            ? attachmentPaths.map((path) => ({ path }))
            : undefined,
        attachmentPathBase,
    });
    if (!ok) {
        fail(`JMAP request failed (HTTP ${status}): ${bodyText}`, 1);
    }
    process.stdout.write(bodyText.endsWith("\n") ? bodyText : bodyText + "\n");
}
async function cmdHelp(argv) {
    let parsed;
    try {
        parsed = parseArgs({
            args: argv,
            options: {
                topic: { type: "string" },
                help: { type: "boolean", short: "h" },
            },
            strict: true,
            allowPositionals: false,
        });
    }
    catch (err) {
        fail(err.message, 2);
    }
    if (parsed.values.help) {
        process.stdout.write(`Usage: atomicmail help [--topic TOPIC]

Topics include: overview, installation, auth, jmap_cheatsheet, tools, presets, troubleshooting, readme.
Topic readme prints the built-in SKILL stub.
`);
        process.exit(0);
    }
    const topic = parsed.values.topic;
    process.stdout.write(await getHelp(topic, "skill") + "\n");
}
async function main() {
    const argv = process.argv.slice(2);
    if (argv.length === 0 || argv[0] === "-h" || argv[0] === "--help") {
        exitUsage(0);
    }
    const cmd = argv[0];
    const rest = argv.slice(1);
    switch (cmd) {
        case "register":
            await cmdRegister(rest);
            break;
        case "jmap_request":
            await cmdJmapRequest(rest);
            break;
        case "help":
            await cmdHelp(rest);
            break;
        default:
            process.stderr.write(`Unknown command: ${cmd}\n\n`);
            process.stdout.write(USAGE);
            process.exit(2);
    }
}
main().catch((err) => {
    fail(err instanceof Error ? err.message : String(err));
});
