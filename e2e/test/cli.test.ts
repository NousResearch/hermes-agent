import { test, expect } from "@microsoft/tui-test";
import {mkdtempSync, rmSync} from "fs"

const CTRL_C = "\x03";

test.describe("Hermes CLI basics", () => {
  const HERMES_HOME = mkdtempSync("hermes-home")
  test.use({
    env: {HERMES_HOME},
  })
  test("hermes command is available and shows version", async ({ terminal }) => {
    terminal.write("hermes --version\n");
    // Wait for the version output to appear
    await expect(terminal.getByText(/hermes/gi, { full: false })).toBeVisible({ timeout: 15000 });
  });

  test("hermes setup wizard starts interactively", async ({ terminal }) => {
    terminal.write("hermes setup\n");
    
    // Wait for the wizard to start (e.g., looking for "Configure Hermes Agent" or similar)
    await expect(terminal.getByText(/configure|setup|wizard|api key/gi)).toBeVisible({ timeout: 15000 });
    
    // Wait for the abort/exit message (KeyboardInterrupt is what python emits on ctrl+c)
    await expect(terminal.getByText(/abort|cancel|exit|terminated|keyboardinterrupt/gi)).toBeVisible({ timeout: 5000 });
  });

  test.afterAll(() => {
    rmSync(HERMES_HOME, { force: true,recursive: true})
  })
});
