import { test, expect, Shell } from "@microsoft/tui-test";
import {mkdtempSync, rmSync} from "fs"

if(process.env.CI === "true") {
  test.describe("install hermes", () => {
    const HERMES_HOME = mkdtempSync("hermes-home")
    test.use({
      shell: Shell.Bash,
      env: {HERMES_HOME},
    })

    test("hermes installer works", async ({ terminal }) => {
      // simulate curl | bash for installer script
      terminal.write("cat $GITHUB_WORKSPACE/install.sh | bash\n");
      // Wait for the version output to appear
      await expect(terminal.getByText(/asdfasdfasdf/gi, { full: false })).toBeVisible({ timeout: 15000 });
    });

    test.afterAll(() => {
      rmSync(HERMES_HOME, { force: true,recursive: true})
    })
  });

}