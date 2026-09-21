import fs from "node:fs/promises";

export async function loadTextFile(filePath) {
  return await fs.readFile(filePath, "utf8");
}
