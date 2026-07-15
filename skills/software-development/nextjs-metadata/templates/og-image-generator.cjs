/**
 * OG Image Generator — creates 1200x630 OpenGraph images via Node.js canvas.
 *
 * Usage: node og-image-generator.cjs --title "My Page" --output public/og-image.png
 *
 * Requires: npm install canvas
 */

const { createCanvas } = require("canvas");
const fs = require("fs");
const path = require("path");

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { title: "My App", subtitle: "", output: "public/og-image.png" };

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--title" && args[i + 1]) opts.title = args[++i];
    if (args[i] === "--subtitle" && args[i + 1]) opts.subtitle = args[++i];
    if (args[i] === "--output" && args[i + 1]) opts.output = args[++i];
  }
  return opts;
}

function generateOgImage({ title, subtitle, output }) {
  const width = 1200;
  const height = 630;
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");

  // Background gradient
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#0f172a");
  gradient.addColorStop(1, "#1e293b");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  // Accent bar
  ctx.fillStyle = "#3b82f6";
  ctx.fillRect(0, height - 8, width, 8);

  // Title
  ctx.fillStyle = "#f8fafc";
  ctx.font = "bold 64px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(title, width / 2, subtitle ? height / 2 - 30 : height / 2);

  // Subtitle
  if (subtitle) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "28px sans-serif";
    ctx.fillText(subtitle, width / 2, height / 2 + 40);
  }

  // Write to file
  const outDir = path.dirname(output);
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const buffer = canvas.toBuffer("image/png");
  fs.writeFileSync(output, buffer);
  console.log(`Generated: ${output} (${buffer.length} bytes)`);
}

const opts = parseArgs();
generateOgImage(opts);
