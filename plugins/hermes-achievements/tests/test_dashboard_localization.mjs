import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const bundle = readFileSync(
  new URL("../dashboard/dist/index.js", import.meta.url),
  "utf8",
);
const start = bundle.indexOf("  function tx(");
const end = bundle.indexOf("\n\n  function localizedMetric", start);

assert.notEqual(start, -1, "dashboard bundle must define tx");
assert.notEqual(end, -1, "dashboard bundle must define localizedMetric");

const sandbox = {};
vm.runInNewContext(
  `${bundle.slice(start, end)}\nthis.localizedAchievement = localizedAchievement;`,
  sandbox,
);

test("hidden secret achievements retain the API-provided name and localize the generic description", () => {
  const achievement = {
    id: "port_3000_taken",
    state: "secret",
    name: "???",
    description: "Secret achievement: hidden until Hermes detects relevant behavior.",
    category: "Debugging Chaos",
  };
  const translations = {
    achievements: {
      catalog: {
        port_3000_taken: {
          name: "3000 埠已有人",
          description: "多次撞見開發伺服器的連接埠衝突。",
        },
      },
      guide: {
        secret_body: "秘密成就會隱藏其確切觸發條件。",
      },
    },
  };

  const localized = sandbox.localizedAchievement(translations, achievement);

  assert.equal(localized.name, "???");
  assert.equal(localized.description, translations.achievements.guide.secret_body);
});
