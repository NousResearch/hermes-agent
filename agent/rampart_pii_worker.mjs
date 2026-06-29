#!/usr/bin/env node

// Rampart is published as a JavaScript package, so Hermes keeps the optional
// redaction backend in a Node worker instead of adding a Python ML stack to the
// base install. `hermes redact setup` installs the pinned npm packages into the
// active Hermes home, then Python invokes this worker as an isolated subprocess.

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

async function loadRampart() {
  const names = ["@nationaldesignstudio/rampart", "rampart"];
  let lastError;
  for (const name of names) {
    try {
      return await import(name);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("Rampart package not found");
}

async function loadTransformers() {
  try {
    return await import("@huggingface/transformers");
  } catch (error) {
    throw new Error(
      "@huggingface/transformers is required for Rampart model-backed redaction in Node"
    );
  }
}

function extractProtectedText(result) {
  if (typeof result === "string") return result;
  if (!result || typeof result !== "object") return null;
  return result.text || result.protectedText || result.protected_text || result.output || null;
}

async function protectWithRampartHeuristics(texts) {
  const mod = await loadRampart();
  const createGuard = mod.createGuard || mod.default?.createGuard;
  if (typeof createGuard !== "function") {
    throw new Error("Rampart package does not expose createGuard()");
  }
  const guard = await createGuard({ heuristicsOnly: true });
  if (!guard || typeof guard.protect !== "function") {
    throw new Error("Rampart guard does not expose protect()");
  }

  const protectedTexts = [];
  for (const text of texts) {
    const result = await guard.protect(String(text));
    const protectedText = extractProtectedText(result);
    if (typeof protectedText !== "string") {
      throw new Error("Rampart protect() returned an unsupported payload");
    }
    protectedTexts.push(protectedText);
  }
  return protectedTexts;
}

async function createRampartGuardWithNodeModel(options) {
  const mod = await loadRampart();
  if (typeof mod.createGuard !== "function" || typeof mod.detectNer !== "function") {
    throw new Error("Rampart package does not expose the required guard and NER APIs");
  }
  const { pipeline, env } = await loadTransformers();
  if (env) {
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
  }
  const classifier = await pipeline("token-classification", options.model, {
    dtype: "q4",
    device: "cpu",
  });
  const adapter = (text, detectOptions) => classifier(text, detectOptions);
  if (classifier.tokenizer && typeof classifier.tokenizer.encode === "function") {
    adapter.countTokens = (text) =>
      classifier.tokenizer.encode(text, { add_special_tokens: false }).length;
  }

  return mod.createGuard({ ner: (text) => mod.detectNer(text, adapter) });
}

async function protectWithGuard(texts, guard) {
  const redactedTexts = [];
  for (const text of texts) {
    const result = await guard.protect(String(text));
    const protectedText = extractProtectedText(result);
    if (typeof protectedText !== "string") {
      throw new Error("Rampart protect() returned an unsupported payload");
    }
    redactedTexts.push(protectedText);
  }
  return redactedTexts;
}

async function main() {
  const input = JSON.parse(await readStdin());
  const texts = Array.isArray(input.texts) ? input.texts : [];
  const options = {
    model: input.model || "nationaldesignstudio/rampart",
    heuristicsOnly: Boolean(input.heuristicsOnly),
  };
  const protectedTexts = options.heuristicsOnly
    ? await protectWithRampartHeuristics(texts)
    : await protectWithGuard(texts, await createRampartGuardWithNodeModel(options));

  process.stdout.write(JSON.stringify({ texts: protectedTexts }));
}

main().catch((error) => {
  const name = error && error.name ? error.name : "Error";
  const message = error && error.message ? `: ${error.message}` : "";
  process.stderr.write(`PII redaction worker failed: ${name}${message}\n`);
  process.exit(1);
});
