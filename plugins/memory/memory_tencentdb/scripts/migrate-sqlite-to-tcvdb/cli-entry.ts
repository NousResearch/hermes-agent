import { runMigrationCli } from "./sqlite-to-tcvdb.js";

const TAG = "[memory-tdai][migrate-cli]";

try {
  const summary = await runMigrationCli(process.argv.slice(2));
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
} catch (err) {
  const message = err instanceof Error ? err.message : String(err);
  process.stderr.write(`${TAG} ${message}\n`);
  process.exitCode = 1;
}
