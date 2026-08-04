/**
 * Benchmark: fastEstimateTokens vs tiktoken cl100k_base
 */
import { fastEstimateTokens } from "../src/offload/fast-token-estimate.ts";
import { getEncoding } from "js-tiktoken";
import { readFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const enc = getEncoding("cl100k_base");

function tiktokenCount(text: string): number {
  return enc.encode(text).length;
}

// Load corpus
const corpusDir = join(__dirname, "../token_count/corpus");
const testTexts: { name: string; text: string }[] = [];

if (existsSync(corpusDir)) {
  const files = ["en_pride.txt", "en_arxiv.txt", "cn_hlm.txt", "cn_sgy.txt", "fr_french.txt",
    "ru_russian.txt", "ja_japanese.txt", "ko_korean.txt", "ar_arabic.txt",
    "de_german.txt", "es_spanish.txt", "pt_portuguese.txt"];
  for (const f of files) {
    const fp = join(corpusDir, f);
    if (existsSync(fp)) {
      testTexts.push({ name: f.replace(".txt", ""), text: readFileSync(fp, "utf-8").slice(0, 100_000) });
    }
  }
}

// Add typical agent scenarios
testTexts.push({ name: "json_messages", text: JSON.stringify([
  { role: "user", content: "Hello world" },
  { role: "assistant", content: "Hi! How can I help?" },
  { role: "user", content: "Run ls -la" },
  { role: "toolResult", toolCallId: "call_123", content: "total 48\ndrwxr-xr-x  5 user user 4096 May 18 10:00 .\n-rw-r--r--  1 user user 1234 May 18 09:30 package.json\n" },
]).repeat(50) });
testTexts.push({ name: "mixed_code_zh", text: "// 这是一个测试函数\nfunction hello(name: string): string {\n  return `你好 ${name}!`;\n}\n".repeat(1000) });

console.log("\n══════════════════════════════════════════════════════════════════");
console.log("  fastEstimateTokens vs tiktoken cl100k_base");
console.log("══════════════════════════════════════════════════════════════════\n");

const header = [
  "文本".padEnd(18), "chars".padStart(8), "tiktoken".padStart(9),
  "estimate".padStart(9), "error".padStart(7), "tk_ms".padStart(7), "est_ms".padStart(7), "speedup".padStart(8),
];
console.log(header.join(" │ "));
console.log("─".repeat(85));

let totalTk = 0, totalEst = 0, totalTkMs = 0, totalEstMs = 0;

for (const { name, text } of testTexts) {
  const t0 = performance.now();
  const tk = tiktokenCount(text);
  const tkMs = performance.now() - t0;

  const t1 = performance.now();
  const est = fastEstimateTokens(text);
  const estMs = performance.now() - t1;

  const err = ((est - tk) / tk * 100).toFixed(1);
  const speedup = (tkMs / Math.max(estMs, 0.01)).toFixed(0);
  const mark = Math.abs(est - tk) / tk <= 0.10 ? "✅" : "❌";

  totalTk += tk; totalEst += est; totalTkMs += tkMs; totalEstMs += estMs;

  console.log([
    name.padEnd(18), text.length.toLocaleString().padStart(8),
    tk.toLocaleString().padStart(9), est.toLocaleString().padStart(9),
    `${err}%`.padStart(7), tkMs.toFixed(1).padStart(7), estMs.toFixed(1).padStart(7),
    `${speedup}x`.padStart(8),
  ].join(" │ ") + ` ${mark}`);
}

console.log("─".repeat(85));
const totalErr = ((totalEst - totalTk) / totalTk * 100).toFixed(1);
console.log([
  "TOTAL".padEnd(18), "".padStart(8),
  totalTk.toLocaleString().padStart(9), totalEst.toLocaleString().padStart(9),
  `${totalErr}%`.padStart(7), totalTkMs.toFixed(0).padStart(7), totalEstMs.toFixed(0).padStart(7),
  `${(totalTkMs / totalEstMs).toFixed(0)}x`.padStart(8),
].join(" │ "));

console.log(`\n  精度: 平均误差 ${totalErr}%`);
console.log(`  速度: tiktoken ${totalTkMs.toFixed(0)}ms vs estimate ${totalEstMs.toFixed(0)}ms (${(totalTkMs / totalEstMs).toFixed(0)}x faster)`);
console.log();
