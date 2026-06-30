/**
 * Demo A — 与生产一致的接入：/v1/chat/completions + SSE 流式。
 *
 * 依赖：npm i openai
 * 运行：npx tsx demo_a.ts
 *
 * 说明：hermes 是 OpenAI 兼容服务端，直接套官方 openai SDK，
 *       SSE 解析全免。model 字段填 "hermes-agent"（占位，真正模型由 hermes 的 Provider 配置决定）。
 */
import OpenAI from "openai";

const BASE = "http://127.0.0.1:8642/v1";
const KEY = process.env.API_SERVER_KEY ?? "demo-key-123"; // 与 API_SERVER_KEY 一致

const client = new OpenAI({ baseURL: BASE, apiKey: KEY });

const stream = await client.chat.completions.create({
  model: "hermes-agent",
  messages: [{ role: "user", content: "用一句话介绍你自己，并说出现在几点" }],
  stream: true,
});

process.stdout.write("assistant: ");
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
console.log();
