import fs from "node:fs";
import path from "node:path";

import { assertDisposableDirectory } from "./contract.mjs";

export function buildTrustBootstrapPlan(configDirectory, workDirectory) {
  const config = assertDisposableDirectory(configDirectory);
  const work = assertDisposableDirectory(workDirectory);
  if (path.dirname(config) !== path.dirname(work)) {
    throw new Error("config and work directories must share one disposable parent");
  }
  if (fs.readdirSync(config).length !== 0) {
    throw new Error("isolated CLAUDE_CONFIG_DIR must be empty before trust bootstrap");
  }
  if (fs.readdirSync(work).length !== 0) {
    throw new Error("disposable work directory must be empty before trust bootstrap");
  }
  const target = path.join(config, ".claude.json");
  return Object.freeze({
    additional_consent_required:
      `Authorize one trust record for the empty disposable Claude workspace ${work}, ` +
      `written only to ${target}, removed with its temporary parent, and never copied to global config?`,
    target,
    before: null,
    after: { projects: { [work]: { hasTrustDialogAccepted: true } } },
    write_mode: "0600",
    cleanup: `remove the shared disposable parent ${path.dirname(config)}`,
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  if (process.argv.length !== 4) {
    throw new Error("usage: node trust-bootstrap-plan.mjs CONFIG_DIR WORK_DIR");
  }
  process.stdout.write(`${JSON.stringify(buildTrustBootstrapPlan(process.argv[2], process.argv[3]), null, 2)}\n`);
}
