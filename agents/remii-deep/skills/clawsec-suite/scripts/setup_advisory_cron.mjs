#!/usr/bin/env node

import { spawnSync as runProcessSync } from "node:child_process";

const JOB_NAME = process.env.CLAWSEC_ADVISORY_CRON_NAME?.trim() || "ClawSec Advisory Scan";
const JOB_EVERY = process.env.CLAWSEC_ADVISORY_CRON_EVERY?.trim() || "6h";
const JOB_DESCRIPTION =
  "Trigger a periodic ClawSec advisory scan in the main session and ask for approval before removing flagged skills.";
const SYSTEM_EVENT =
  "Run ClawSec advisory scan. If installed skills are flagged as malicious or removal is recommended, notify the user and request explicit approval before any removal.";

function sh(cmd, args) {
  const result = runProcessSync(cmd, args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });

  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    const details = (result.stderr || result.stdout || "").trim();
    throw new Error(`${cmd} ${args.join(" ")} failed${details ? `: ${details}` : ""}`);
  }

  return result.stdout;
}

function requireOpenClawCli() {
  try {
    sh("openclaw", ["--version"]);
  } catch (error) {
    throw new Error(
      "openclaw CLI is required. Install OpenClaw and ensure `openclaw` is available in PATH. " +
        `Original error: ${String(error)}`,
      { cause: error },
    );
  }
}

function findExistingJobId(jobsPayload) {
  if (!jobsPayload || !Array.isArray(jobsPayload.jobs)) return null;
  const existing = jobsPayload.jobs.find((job) => job && job.name === JOB_NAME);
  return existing?.id ?? null;
}

function addJob() {
  const out = sh("openclaw", [
    "cron",
    "add",
    "--name",
    JOB_NAME,
    "--description",
    JOB_DESCRIPTION,
    "--every",
    JOB_EVERY,
    "--session",
    "main",
    "--system-event",
    SYSTEM_EVENT,
    "--wake",
    "now",
    "--json",
  ]);

  try {
    const payload = JSON.parse(out);
    return payload?.id ?? null;
  } catch {
    return null;
  }
}

function editJob(jobId) {
  sh("openclaw", [
    "cron",
    "edit",
    jobId,
    "--name",
    JOB_NAME,
    "--description",
    JOB_DESCRIPTION,
    "--enable",
    "--every",
    JOB_EVERY,
    "--session",
    "main",
    "--system-event",
    SYSTEM_EVENT,
    "--wake",
    "now",
  ]);
}

function printPreflightSummary() {
  const lines = [
    "Preflight review:",
    "- This setup creates or updates an unattended openclaw cron job in the main session.",
    "- Required runtime: openclaw CLI, node.",
    `- Schedule: every ${JOB_EVERY}`,
    "- The system event triggers an advisory scan and must request explicit approval before any removal.",
  ];

  process.stdout.write(lines.join("\n") + "\n\n");
}

function main() {
  requireOpenClawCli();
  printPreflightSummary();

  const jobsOut = sh("openclaw", ["cron", "list", "--json"]);
  const jobsPayload = JSON.parse(jobsOut);
  const existingJobId = findExistingJobId(jobsPayload);

  if (existingJobId) {
    editJob(existingJobId);
    process.stdout.write(`Updated cron job ${existingJobId}: ${JOB_NAME}\n`);
  } else {
    const createdId = addJob();
    if (createdId) {
      process.stdout.write(`Created cron job ${createdId}: ${JOB_NAME}\n`);
    } else {
      process.stdout.write(`Created cron job: ${JOB_NAME}\n`);
    }
  }

  process.stdout.write(`Schedule: every ${JOB_EVERY}\n`);
  process.stdout.write("Session target: main (system event + wake now)\n");
}

try {
  main();
} catch (error) {
  process.stderr.write(`${String(error)}\n`);
  process.exit(1);
}
