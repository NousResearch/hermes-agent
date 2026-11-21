import assert from "node:assert/strict";
import test from "node:test";

import { createSpectrumRuntime } from "../spectrum-runtime.mjs";

function runtimeHarness() {
  const imports = [];
  const configs = [];
  const localIMessage = Object.assign(
    () => ({ provider: "local-runtime" }),
    {
      config: () => ({ provider: "local" }),
      effect: { message: { confetti: "local-confetti" } },
    }
  );
  const imessage = Object.assign(
    () => ({ provider: "cloud-runtime" }),
    {
      config: () => ({ provider: "cloud" }),
      effect: { message: { confetti: "cloud-confetti" } },
    }
  );
  const localEffect = Symbol("local-effect");
  const cloudEffect = Symbol("cloud-effect");
  const core = {
    Spectrum: async (config) => {
      configs.push(config);
      return { messages: [] };
    },
    attachment: Symbol("attachment"),
    voice: Symbol("voice"),
    text: Symbol("text"),
    markdown: Symbol("markdown"),
    richlink: Symbol("richlink"),
    typing: Symbol("typing"),
    poll: Symbol("poll"),
  };
  const modules = {
    "@spectrum-ts/core": core,
    "@spectrum-ts/imessage-local": { localIMessage, effect: localEffect },
    "spectrum-ts/providers/imessage": { imessage, effect: cloudEffect },
  };
  const importer = async (specifier) => {
    imports.push(specifier);
    return modules[specifier];
  };
  return {
    cloudEffect,
    configs,
    core,
    importer,
    imports,
    localEffect,
    localIMessage,
    imessage,
  };
}

test("local mode selects the dedicated local provider without cloud credentials", async () => {
  const harness = runtimeHarness();

  const runtime = await createSpectrumRuntime({
    localMode: true,
    projectId: "unused-id",
    projectSecret: "unused-secret",
    telemetry: false,
    importer: harness.importer,
  });

  assert.deepEqual(harness.imports, [
    "@spectrum-ts/core",
    "@spectrum-ts/imessage-local",
  ]);
  assert.deepEqual(harness.configs, [{
    providers: [{ provider: "local" }],
    options: { flattenGroups: true },
    telemetry: false,
  }]);
  assert.equal(runtime.attachment, harness.core.attachment);
  assert.equal(runtime.provider, harness.localIMessage);
  assert.equal(runtime.spectrumRichlink, harness.core.richlink);
  assert.equal(runtime.spectrumPoll, harness.core.poll);
  assert.equal(runtime.imessageEffect, harness.localEffect);
  assert.deepEqual(runtime.messageEffects, { confetti: "local-confetti" });
});

test("cloud mode selects the managed provider and passes project credentials", async () => {
  const harness = runtimeHarness();

  const runtime = await createSpectrumRuntime({
    localMode: false,
    projectId: "project-id",
    projectSecret: "project-secret",
    telemetry: true,
    importer: harness.importer,
  });

  assert.deepEqual(harness.imports, [
    "@spectrum-ts/core",
    "spectrum-ts/providers/imessage",
  ]);
  assert.deepEqual(harness.configs, [{
    providers: [{ provider: "cloud" }],
    options: { flattenGroups: true },
    telemetry: true,
    projectId: "project-id",
    projectSecret: "project-secret",
  }]);
  assert.equal(runtime.provider, harness.imessage);
  assert.equal(runtime.imessageEffect, harness.cloudEffect);
  assert.deepEqual(runtime.messageEffects, { confetti: "cloud-confetti" });
});

test("installed Spectrum packages expose both provider APIs", async () => {
  const [core, local, cloud] = await Promise.all([
    import("@spectrum-ts/core"),
    import("@spectrum-ts/imessage-local"),
    import("spectrum-ts/providers/imessage"),
  ]);

  assert.equal(typeof core.Spectrum, "function");
  assert.equal(local.localIMessage.config().__name, "local_imessage");
  assert.equal(cloud.imessage.config().__name, "imessage");
});
