import { existsSync, readdirSync, readFileSync } from "fs";
import { join, basename } from "path";
import { PDFDocument } from "pdf-lib";

interface PageInfo {
  filename: string;
  path: string;
  index: number;
  promptPath?: string;
}

function parseArgs(): { dir: string; output?: string } {
  const args = process.argv.slice(2);
  let dir = "";
  let output: string | undefined;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--output" || args[i] === "-o") {
      output = args[++i];
    } else if (!args[i].startsWith("-")) {
      dir = args[i];
    }
  }

  if (!dir) {
    console.error("Usage: bun merge-to-pdf.ts <comic-dir> [--output filename.pdf]");
    process.exit(1);
  }

  return { dir, output };
}

function findComicPages(dir: string): PageInfo[] {
  if (!existsSync(dir)) {
    console.error(`Directory not found: ${dir}`);
    process.exit(1);
  }

  const files = readdirSync(dir);
  const pagePattern = /^(\d+)-(cover|page)(-[\w-]+)?\.(png|jpg|jpeg)$/i;
  const promptsDir = join(dir, "prompts");
  const hasPrompts = existsSync(promptsDir);

  const pages: PageInfo[] = files
    .filter((f) => pagePattern.test(f))
    .map((f) => {
      const match = f.match(pagePattern);
      const baseName = f.replace(/\.(png|jpg|jpeg)$/i, "");
      const promptPath = hasPrompts ? join(promptsDir, `${baseName}.md`) : undefined;

      return {
        filename: f,
        path: join(dir, f),
        index: parseInt(match![1], 10),
        promptPath: promptPath && existsSync(promptPath) ? promptPath : undefined,
      };
    })
    .sort((a, b) => a.index - b.index);

  if (pages.length === 0) {
    console.error(`No comic pages found in: ${dir}`);
    console.error("Expected format: 00-cover-slug.png, 01-page-slug.png, etc.");
    process.exit(1);
  }

  return pages;
}

async function createPdf(pages: PageInfo[], outputPath: string) {
  const pdfDoc = await PDFDocument.create();
  pdfDoc.setAuthor("baoyu-comic");
  pdfDoc.setSubject("Generated Comic");

  for (const page of pages) {
    const imageData = readFileSync(page.path);
    const ext = page.filename.toLowerCase();
    const image = ext.endsWith(".png")
      ? await pdfDoc.embedPng(imageData)
      : await pdfDoc.embedJpg(imageData);

    const { width, height } = image;
    const pdfPage = pdfDoc.addPage([width, height]);

    pdfPage.drawImage(image, {
      x: 0,
      y: 0,
      width,
      height,
    });

    console.log(`Added: ${page.filename}${page.promptPath ? " (prompt available)" : ""}`);
  }

  const pdfBytes = await pdfDoc.save();
  await Bun.write(outputPath, pdfBytes);

  console.log(`\nCreated: ${outputPath}`);
  console.log(`Total pages: ${pages.length}`);
}

async function main() {
  const { dir, output } = parseArgs();
  const pages = findComicPages(dir);

  const dirName = basename(dir) === "comic" ? basename(join(dir, "..")) : basename(dir);
  const outputPath = output || join(dir, `${dirName}.pdf`);

  console.log(`Found ${pages.length} pages in: ${dir}\n`);

  await createPdf(pages, outputPath);
}

main().catch((err) => {
  console.error("Error:", err.message);
  process.exit(1);
});
