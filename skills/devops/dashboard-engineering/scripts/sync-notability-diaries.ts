#!/usr/bin/env bun

/**
 * Notability Diary → hermes.db
 *
 * Checks Google Drive Notability/Diary folder for new PDFs.
 * Downloads, converts to PNG, extracts text via OpenRouter vision,
 * and stores as observations in hermes.db.
 *
 * Uses Composio REST API (not CLI) for Google Drive operations.
 */

import {
  existsSync,
  mkdirSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { getDb } from "./db.js";

// ── Config ───────────────────────────────────────────────────────────

const NOTABILITY_DIARY_FOLDER_ID = "1KiJ0sb96cRqypNCfgCksQnQ93AryX2PZ";
const TMP_DIR = join(homedir(), "tmp", "diary-sync");
const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY || "";
const COMPOSIO_API_KEY = process.env.COMPOSIO_API_KEY || "";
const COMPOSIO_BASE = "https://backend.composio.dev";

// ── Composio REST Helpers ────────────────────────────────────────────

async function composioExecute(tool: string, args: Record<string, any>): Promise<any> {
  if (!COMPOSIO_API_KEY) throw new Error("COMPOSIO_API_KEY not set");

  const resp = await fetch(`${COMPOSIO_BASE}/api/v2/connectedAccounts`, {
    headers: {
      "x-api-key": COMPOSIO_API_KEY,
      "Content-Type": "application/json",
    },
  });
  const accounts = await resp.json() as any;
  const googleAccount = accounts.items?.find(
    (a: any) => a.appName === "googledrive" && a.status === "ACTIVE"
  );
  if (!googleAccount) throw new Error("No active Google Drive connection");

  const execResp = await fetch(`${COMPOSIO_BASE}/api/v2/actions/${tool}/execute`, {
    method: "POST",
    headers: {
      "x-api-key": COMPOSIO_API_KEY,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      connectedAccountId: googleAccount.id,
      input: args,
    }),
  });
  const result = await execResp.json() as any;
  if (!result.successful) {
    throw new Error(`Composio ${tool} failed: ${JSON.stringify(result.error || result)}`);
  }
  return result.data;
}

async function listFolder(
  folderId: string,
): Promise<Array<{ id: string; name: string; mimeType: string }>> {
  const data = await composioExecute("GOOGLEDRIVE_LIST_CHILDREN_V2", { folderId });
  const items = data.items || [];

  const results: Array<{ id: string; name: string; mimeType: string }> = [];
  for (const item of items) {
    const meta = await composioExecute("GOOGLEDRIVE_GET_FILE_METADATA", {
      fileId: item.id,
    });
    results.push({
      id: meta.id,
      name: meta.name || "unknown",
      mimeType: meta.mimeType || "unknown",
    });
  }
  return results;
}

async function downloadFile(fileId: string, destPath: string): Promise<void> {
  const data = await composioExecute("GOOGLEDRIVE_DOWNLOAD_FILE", { fileId });
  const s3url = data.downloaded_file_content?.s3url;
  if (!s3url) throw new Error("No S3 URL in download response");

  const urlFile = `${destPath}.url`;
  writeFileSync(urlFile, s3url);
  const { execSync } = await import("node:child_process");
  execSync(`curl -sL -o "${destPath}" "$(cat '${urlFile}')"`, {
    timeout: 60_000,
  });
  unlinkSync(urlFile);

  if (!existsSync(destPath) || readFileSync(destPath).length < 100) {
    throw new Error("Downloaded file is missing or too small");
  }
}

// ── Vision Extraction ────────────────────────────────────────────────

async function extractTextFromImage(imagePath: string): Promise<string> {
  if (!OPENROUTER_KEY) throw new Error("OPENROUTER_API_KEY not set");

  const imageBytes = readFileSync(imagePath);
  const base64 = imageBytes.toString("base64");
  const ext = imagePath.endsWith(".png") ? "image/png" : "image/jpeg";

  const resp = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENROUTER_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "google/gemini-2.5-flash",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: { url: `data:${ext};base64,${base64}` },
            },
            {
              type: "text",
              text: "Read ALL handwriting on this diary page. Transcribe everything exactly — every word, bullet point, task, and note. Preserve structure. If the page is blank, respond with exactly: [BLANK PAGE]",
            },
          ],
        },
      ],
      max_tokens: 4096,
    }),
  });

  const result = (await resp.json()) as any;
  const text = result.choices?.[0]?.message?.content || "";
  return text.trim();
}

// ── PDF Processing ───────────────────────────────────────────────────

function pdfToPng(pdfPath: string, outputDir: string): string[] {
  const { execSync } = require("node:child_process");
  execSync(`pdftoppm -png -r 300 "${pdfPath}" "${outputDir}/page"`, {
    timeout: 30_000,
  });

  const files: string[] = [];
  for (let i = 1; i <= 100; i++) {
    const p = `${outputDir}/page-${String(i).padStart(2, "0")}.png`;
    const p2 = `${outputDir}/page-${i}.png`;
    if (existsSync(p)) files.push(p);
    else if (existsSync(p2)) files.push(p2);
    else break;
  }
  return files;
}

// ── Main ─────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes("--dry-run");
  const reAll = args.includes("--all");

  const db = getDb();
  mkdirSync(TMP_DIR, { recursive: true });

  console.log("Notability Diary → hermes.db\n");

  // Get existing diary observations to avoid re-processing
  const existing = new Set<string>();
  const rows = db
    .query("SELECT source_ref FROM observations WHERE type = 'diary'")
    .all() as Array<{ source_ref: string }>;
  for (const r of rows) {
    if (r.source_ref) existing.add(r.source_ref);
  }
  console.log(`Already ingested: ${existing.size} diaries`);

  // List Diary folder
  console.log(`\nFetching Diary folder (${NOTABILITY_DIARY_FOLDER_ID})...`);
  const files = await listFolder(NOTABILITY_DIARY_FOLDER_ID);
  const pdfs = files.filter((f) => f.mimeType === "application/pdf");
  console.log(`Found ${pdfs.length} PDFs`);

  let ingested = 0;

  for (const pdf of pdfs) {
    const alreadyDone = existing.has(pdf.id);
    if (alreadyDone && !reAll) {
      console.log(`  ⏭  ${pdf.name} (already ingested)`);
      continue;
    }

    console.log(`\n  📄 ${pdf.name}`);

    if (dryRun) {
      console.log(`     [dry-run] Would download + extract`);
      continue;
    }

    try {
      // Download PDF
      const pdfPath = join(TMP_DIR, pdf.name);
      await downloadFile(pdf.id, pdfPath);
      console.log(
        `     ↓ downloaded (${(readFileSync(pdfPath).length / 1024).toFixed(0)}KB)`,
      );

      // Convert to PNGs
      const pageDir = join(TMP_DIR, pdf.name.replace(".pdf", ""));
      mkdirSync(pageDir, { recursive: true });
      const pages = pdfToPng(pdfPath, pageDir);
      console.log(`     🖼  ${pages.length} page(s)`);

      // Extract text from each page
      const pageTexts: string[] = [];
      for (let i = 0; i < pages.length; i++) {
        const text = await extractTextFromImage(pages[i]);
        if (!text.includes("[BLANK PAGE]")) {
          pageTexts.push(text);
          console.log(`     ✅ Page ${i + 1}: ${text.length} chars`);
        } else {
          console.log(`     ⬜ Page ${i + 1}: blank`);
        }
      }

      const fullText = pageTexts.join("\n\n---\n\n");
      if (!fullText.trim()) {
        console.log(`     ⚠️  No content extracted, skipping`);
        continue;
      }

      // Parse date from filename: "Note Jul 21, 2026.pdf" → "2026-07-21"
      const dateMatch = pdf.name.match(/(\w+)\s+(\d+),\s+(\d{4})/);
      let diaryDate = new Date().toISOString().slice(0, 10);
      if (dateMatch) {
        const monthMap: Record<string, string> = {
          Jan: "01", Feb: "02", Mar: "03", Apr: "04",
          May: "05", Jun: "06", Jul: "07", Aug: "08",
          Sep: "09", Oct: "10", Nov: "11", Dec: "12",
        };
        const mon = monthMap[dateMatch[1]] || "01";
        const day = dateMatch[2].padStart(2, "0");
        diaryDate = `${dateMatch[3]}-${mon}-${day}`;
      }

      // Store in hermes.db
      const entityId = `diary-${diaryDate}`;
      db.query(`
        INSERT OR IGNORE INTO entities (id, type, name, description, source_id, created_at, updated_at, confidence)
        VALUES (?, 'diary', ?, ?, 'notability', ?, ?, 1.0)
      `).run(
        entityId,
        `Diary ${diaryDate}`,
        `Notability diary entry from ${diaryDate}`,
        `${diaryDate}T12:00:00`,
        `${diaryDate}T12:00:00`,
      );

      const obsId = `diary-${pdf.id.slice(0, 12)}`;
      db.query(`
        INSERT OR REPLACE INTO observations (id, entity_id, type, content, timestamp, source, source_ref)
        VALUES (?, ?, 'diary', ?, ?, 'notability', ?)
      `).run(obsId, entityId, fullText, `${diaryDate}T12:00:00`, pdf.id);

      ingested++;
      console.log(`     💾 stored in hermes.db (obs: ${obsId})`);

      // Cleanup temp files
      for (const p of pages) {
        try { unlinkSync(p); } catch {}
      }
      try { unlinkSync(pdfPath); } catch {}
    } catch (err: any) {
      console.error(`     ❌ Error: ${err.message}`);
    }
  }

  console.log(`\n✓ Done. ${ingested} new diaries ingested.`);

  // Summary query
  const total = db
    .query("SELECT COUNT(*) as c FROM observations WHERE type = 'diary'")
    .get() as { c: number };
  console.log(`  Total diary observations: ${total.c}`);

  db.close();
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
