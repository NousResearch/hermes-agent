import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(__dirname, "dist");
const profileFiles = {
  front: "front.tokens.json",
  admin: "admin.tokens.json"
};

const genericFontFamilies = new Set([
  "serif",
  "sans-serif",
  "monospace",
  "system-ui",
  "cursive",
  "fantasy",
  "ui-serif",
  "ui-sans-serif",
  "ui-monospace",
  "ui-rounded"
]);

const arg = process.argv[2] ?? "all";
const selectedProfiles =
  arg === "all" ? Object.keys(profileFiles) : Object.keys(profileFiles).filter((name) => name === arg);

if (selectedProfiles.length === 0) {
  throw new Error(`Unknown profile "${arg}". Use "front", "admin", or "all".`);
}

await mkdir(outDir, { recursive: true });

const core = await readJson("core.tokens.json");
let effects = {};
try {
  effects = await readJson("effects.tokens.json"); // ชั้น expressive (optional)
  console.log("Expressive layer: effects.tokens.json merged");
} catch { /* ไม่มีไฟล์ = ข้าม */ }
const coreWithEffects = deepMerge(core, effects);
const builtProfiles = {};

for (const profileName of selectedProfiles) {
  const profile = await readJson(profileFiles[profileName]);
  const built = buildProfile(coreWithEffects, profile, profileName);
  builtProfiles[profileName] = built.tsExport;

  await writeFile(path.join(outDir, `${profileName}.css`), built.css, "utf8");
  await writeFile(path.join(outDir, `${profileName}.tokens.ts`), built.tsModule, "utf8");

  checkContrast(profileName, Object.values(built.tsExport.modes.light));
}

await writeFile(path.join(outDir, "tokens.ts"), renderIndexModule(builtProfiles), "utf8");

console.log(`Generated ${selectedProfiles.length} profile(s): ${selectedProfiles.join(", ")}`);
console.log(`Output: ${path.relative(process.cwd(), outDir)}`);

async function readJson(fileName) {
  const filePath = path.join(__dirname, fileName);
  return JSON.parse(await readFile(filePath, "utf8"));
}

function buildProfile(coreTokens, profileTokens, profileName) {
  const baseProfile = omitKeys(profileTokens, ["theme"]);
  const lightTree = deepMerge(coreTokens, baseProfile);
  const darkProfile = deepMerge(baseProfile, profileTokens.theme?.dark ?? {});
  const darkTree = deepMerge(coreTokens, darkProfile);

  const lightTokens = collectResolvedTokens(lightTree);
  const darkTokens = collectResolvedTokens(darkTree);

  const css = renderCss(profileName, lightTokens, darkTokens);
  const tsExport = {
    profile: profileName,
    modes: {
      light: toTokenMap(lightTokens),
      dark: toTokenMap(darkTokens)
    }
  };
  const tsModule = renderProfileModule(tsExport);

  return { css, tsExport, tsModule };
}

function deepMerge(base, override) {
  if (Array.isArray(base) || Array.isArray(override)) {
    return structuredClone(override ?? base);
  }

  if (!isPlainObject(base) || !isPlainObject(override)) {
    return structuredClone(override ?? base);
  }

  const result = structuredClone(base);
  for (const [key, value] of Object.entries(override)) {
    result[key] = key in result ? deepMerge(result[key], value) : structuredClone(value);
  }
  return result;
}

function omitKeys(input, keys) {
  const blocked = new Set(keys);
  return Object.fromEntries(Object.entries(input).filter(([key]) => !blocked.has(key)));
}

function collectResolvedTokens(tree) {
  const tokens = [];

  walk(tree, [], undefined);
  return tokens;

  function walk(node, tokenPath, inheritedType) {
    if (!isPlainObject(node)) return;

    const nextType = typeof node.$type === "string" ? node.$type : inheritedType;
    if (Object.hasOwn(node, "$value")) {
      const resolved = resolveToken(tree, tokenPath, []);
      tokens.push({
        path: tokenPath.join("."),
        cssName: `--${tokenPath.map(kebab).join("-")}`,
        type: resolved.type,
        value: resolved.value,
        cssValue: toCssValue(resolved.type, resolved.value)
      });
      return;
    }

    for (const [key, value] of Object.entries(node)) {
      if (key.startsWith("$") || key === "theme") continue;
      walk(value, [...tokenPath, key], nextType);
    }
  }
}

function resolveToken(tree, tokenPath, stack) {
  const key = tokenPath.join(".");
  if (stack.includes(key)) {
    throw new Error(`Circular token reference: ${[...stack, key].join(" -> ")}`);
  }

  const token = getByPath(tree, tokenPath);
  if (!isPlainObject(token) || !Object.hasOwn(token, "$value")) {
    throw new Error(`Token not found: ${key}`);
  }

  let type = token.$type ?? inheritedTypeForPath(tree, tokenPath);
  if (!type && isAlias(token.$value)) {
    type = resolveToken(tree, aliasToPath(token.$value), [...stack, key]).type;
  }
  if (!type) {
    throw new Error(`Missing $type for token: ${key}`);
  }

  return {
    type,
    value: resolveValue(tree, token.$value, [...stack, key])
  };
}

function resolveValue(tree, value, stack) {
  if (isAlias(value)) {
    return resolveToken(tree, aliasToPath(value), stack).value;
  }

  if (Array.isArray(value)) {
    return value.map((item) => resolveValue(tree, item, stack));
  }

  if (isPlainObject(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, resolveValue(tree, item, stack)])
    );
  }

  return value;
}

function inheritedTypeForPath(tree, tokenPath) {
  for (let index = tokenPath.length - 1; index >= 0; index -= 1) {
    const parent = getByPath(tree, tokenPath.slice(0, index));
    if (isPlainObject(parent) && typeof parent.$type === "string") {
      return parent.$type;
    }
  }
  return undefined;
}

function getByPath(tree, tokenPath) {
  return tokenPath.reduce((node, key) => (isPlainObject(node) ? node[key] : undefined), tree);
}

function isAlias(value) {
  return typeof value === "string" && /^\{[^{}]+\}$/.test(value);
}

function aliasToPath(alias) {
  return alias.slice(1, -1).split(".");
}

function toCssValue(type, value) {
  switch (type) {
    case "color":
      return colorToCss(value);
    case "dimension":
    case "duration":
      return `${value.value}${value.unit}`;
    case "fontFamily":
      return Array.isArray(value) ? value.map(formatFontFamily).join(", ") : formatFontFamily(value);
    case "fontWeight":
      return String(value);
    case "cubicBezier":
      return `cubic-bezier(${value.join(", ")})`;
    case "shadow":
      return shadowToCss(value);
    case "gradient":
      return gradientToCss(value);
    case "number":
      return String(value);
    default:
      return Array.isArray(value) || isPlainObject(value) ? JSON.stringify(value) : String(value);
  }
}

function colorToCss(value) {
  if (!isPlainObject(value) || value.colorSpace !== "oklch") {
    throw new Error(`Only OKLCH color values are supported. Received: ${JSON.stringify(value)}`);
  }

  const [lightness, chroma, hue] = value.components;
  const alpha = value.alpha ?? 1;
  const base = `oklch(${round(lightness * 100)}% ${round(chroma)} ${round(hue)}`;
  return alpha === 1 ? `${base})` : `${base} / ${round(alpha)})`;
}

function gradientToCss(value) {
  // DTCG gradient: { angle? (extension), stops: [{color, position}] } หรือ array ของ stops ตรง ๆ
  const stops = Array.isArray(value) ? value : value.stops;
  const angle = (!Array.isArray(value) && value.angle) || "135deg";
  const parts = stops.map((stop) => {
    const color = colorToCss(stop.color);
    return stop.position !== undefined ? `${color} ${Math.round(stop.position * 100)}%` : color;
  });
  return `linear-gradient(${angle}, ${parts.join(", ")})`;
}

function shadowToCss(value) {
  const layers = Array.isArray(value) ? value : [value];
  return layers
    .map((layer) => {
      const parts = [
        layer.inset ? "inset" : null,
        dimToCss(layer.offsetX),
        dimToCss(layer.offsetY),
        dimToCss(layer.blur),
        dimToCss(layer.spread),
        layer.color ? colorToCss(layer.color) : null
      ].filter((part) => part !== null && part !== undefined);
      return parts.join(" ");
    })
    .join(", ");
}

function dimToCss(dimension) {
  if (dimension === null || dimension === undefined) return "0";
  if (isPlainObject(dimension) && "value" in dimension) {
    return `${dimension.value}${dimension.unit ?? "px"}`;
  }
  return String(dimension);
}

function oklchToLinearSrgb(color) {
  const [lightness, chroma, hue] = color.components;
  const h = (hue * Math.PI) / 180;
  const a = chroma * Math.cos(h);
  const b = chroma * Math.sin(h);
  const l_ = lightness + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = lightness - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = lightness - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ ** 3;
  const m = m_ ** 3;
  const s = s_ ** 3;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s
  ];
}

function relativeLuminance(color) {
  const [r, g, b] = oklchToLinearSrgb(color).map((v) => Math.min(1, Math.max(0, v)));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrastRatio(colorA, colorB) {
  const la = relativeLuminance(colorA);
  const lb = relativeLuminance(colorB);
  const hi = Math.max(la, lb);
  const lo = Math.min(la, lb);
  return (hi + 0.05) / (lo + 0.05);
}

function checkContrast(profileName, tokens) {
  const byPath = new Map(
    tokens.filter((token) => token.type === "color").map((token) => [token.path, token.value])
  );
  const pairs = [
    ["semantic.color.text.default", "semantic.color.background.canvas"],
    ["semantic.color.text.default", "semantic.color.background.surface"],
    ["semantic.color.text.muted", "semantic.color.background.canvas"]
  ];
  for (const [fg, bg] of pairs) {
    const a = byPath.get(fg);
    const b = byPath.get(bg);
    if (!isPlainObject(a) || !isPlainObject(b)) continue;
    const ratio = contrastRatio(a, b);
    const label = `${profileName}: ${fg.split(".").pop()} on ${bg.split(".").pop()}`;
    if (ratio < 4.5) {
      console.warn(`  [contrast WARN] ${label} = ${ratio.toFixed(2)}:1 (<4.5 · WCAG AA ตก)`);
    } else {
      console.log(`  [contrast OK]   ${label} = ${ratio.toFixed(2)}:1`);
    }
  }
}

function formatFontFamily(value) {
  if (genericFontFamilies.has(value)) return value;
  if (/^[a-zA-Z0-9-]+$/.test(value)) return value;
  return `"${String(value).replaceAll('"', '\\"')}"`;
}

function renderCss(profileName, lightTokens, darkTokens) {
  const lines = [
    "/* Generated by build-tokens.mjs. Do not edit this file directly. */",
    `/* Profile: ${profileName} */`,
    "",
    ":root {",
    ...lightTokens.map((token) => `  ${token.cssName}: ${token.cssValue};`),
    "}",
    "",
    "[data-theme=\"dark\"] {",
    ...darkTokens.map((token) => `  ${token.cssName}: ${token.cssValue};`),
    "}",
    ""
  ];
  return lines.join("\n");
}

function renderProfileModule(tsExport) {
  return [
    "/* Generated by build-tokens.mjs. Do not edit this file directly. */",
    "export type TokenMode = \"light\" | \"dark\";",
    "export type ResolvedToken = {",
    "  readonly path: string;",
    "  readonly type: string;",
    "  readonly value: unknown;",
    "  readonly cssValue: string;",
    "};",
    "",
    `export const tokens = ${JSON.stringify(tsExport, null, 2)} as const;`,
    "export type Tokens = typeof tokens;",
    "export default tokens;",
    ""
  ].join("\n");
}

function renderIndexModule(profiles) {
  return [
    "/* Generated by build-tokens.mjs. Do not edit this file directly. */",
    "export type TokenMode = \"light\" | \"dark\";",
    "export type ProfileName = keyof typeof tokens;",
    "export type ResolvedToken = {",
    "  readonly path: string;",
    "  readonly type: string;",
    "  readonly value: unknown;",
    "  readonly cssValue: string;",
    "};",
    "",
    `export const tokens = ${JSON.stringify(profiles, null, 2)} as const;`,
    "export type Tokens = typeof tokens;",
    "export default tokens;",
    ""
  ].join("\n");
}

function toTokenMap(tokens) {
  return Object.fromEntries(
    tokens.map((token) => [
      token.path,
      {
        path: token.path,
        type: token.type,
        value: token.value,
        cssValue: token.cssValue
      }
    ])
  );
}

function kebab(value) {
  return String(value)
    .replace(/([a-z0-9])([A-Z])/g, "$1-$2")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .toLowerCase();
}

function round(value) {
  return Number.parseFloat(Number(value).toFixed(4));
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
