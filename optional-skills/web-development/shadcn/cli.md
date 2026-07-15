# shadcn CLI Reference

Configuration is read from `components.json`.

> **IMPORTANT:** Always run commands using the project's package runner: `npx shadcn@latest`, `pnpm dlx shadcn@latest`, or `bunx --bun shadcn@latest`. Check `packageManager` from project context to choose the right one. Examples below use `npx shadcn@latest` but substitute the correct runner for the project.

> **IMPORTANT:** Only use the flags documented below. Do not invent or guess flags — if a flag isn't listed here, it doesn't exist. The CLI auto-detects the package manager from the project's lockfile; there is no `--package-manager` flag.

## Contents

- Commands: init, apply, add (dry-run, smart merge), search, view, docs, info, build
- Templates: next, vite, start, react-router, astro
- Presets: named, code, URL formats and fields
- Switching presets

---

## Commands

### `init` — Initialize or create a project

```bash
npx shadcn@latest init [components...] [options]
```

Initializes shadcn/ui in an existing project or creates a new project (when `--name` is provided). Optionally installs components in the same step.

| Flag                    | Short | Description                                               | Default |
| ----------------------- | ----- | --------------------------------------------------------- | ------- |
| `--template <template>` | `-t`  | Template (next, start, vite, next-monorepo, react-router) | —       |
| `--preset [name]`       | `-p`  | Preset configuration (named, code, or URL)                | —       |
| `--yes`                 | `-y`  | Skip confirmation prompt                                  | `true`  |
| `--defaults`            | `-d`  | Use defaults (`--template=next --preset=base-nova`)       | `false` |
| `--force`               | `-f`  | Force overwrite existing configuration                    | `false` |
| `--cwd <cwd>`           | `-c`  | Working directory                                         | current |
| `--name <name>`         | `-n`  | Name for new project                                      | —       |
| `--silent`              | `-s`  | Mute output                                               | `false` |
| `--rtl`                 |       | Enable RTL support                                        | —       |
| `--reinstall`           |       | Re-install existing UI components                         | `false` |
| `--monorepo`            |       | Scaffold a monorepo project                               | —       |
| `--no-monorepo`         |       | Skip the monorepo prompt                                  | —       |

`npx shadcn@latest create` is an alias for `npx shadcn@latest init`.

### `apply` — Apply a preset to an existing project

```bash
npx shadcn@latest apply [preset] [options]
```

Applies a preset to an existing project, overwriting preset-driven config, fonts, CSS variables, and detected UI components.

| Flag                | Short | Description                                | Default |
| ------------------- | ----- | ------------------------------------------ | ------- |
| `--preset <preset>` | —     | Preset configuration (named, code, or URL) | —       |
| `--yes`             | `-y`  | Skip confirmation prompt                   | `false` |
| `--cwd <cwd>`       | `-c`  | Working directory                          | current |
| `--silent`          | `-s`  | Mute output                                | `false` |

`[preset]` is a shorthand for `--preset <preset>`. If both are provided, they must match.
If no preset is provided, the CLI offers to open the custom preset builder on `ui.shadcn.com/create`.

### `add` — Add components

> **IMPORTANT:** To compare local components against upstream or to preview changes, ALWAYS use `npx shadcn@latest add <component> --dry-run`, `--diff`, or `--view`. NEVER fetch raw files from GitHub or other sources manually. The CLI handles registry resolution, file paths, and CSS diffing automatically.

```bash
npx shadcn@latest add [components...] [options]
```

Accepts component names, registry-prefixed names (`@magicui/shimmer-button`),
GitHub item addresses (`owner/repo/item`), URLs, or local paths.

| Flag            | Short | Description                                                                                                          | Default |
| --------------- | ----- | -------------------------------------------------------------------------------------------------------------------- | ------- |
| `--yes`         | `-y`  | Skip confirmation prompt                                                                                             | `false` |
| `--overwrite`   | `-o`  | Overwrite existing files                                                                                             | `false` |
| `--cwd <cwd>`   | `-c`  | Working directory                                                                                                    | current |
| `--all`         | `-a`  | Add all available components                                                                                         | `false` |
| `--path <path>` | `-p`  | Target path for the component                                                                                        | —       |
| `--silent`      | `-s`  | Mute output                                                                                                          | `false` |
| `--dry-run`     |       | Preview all changes without writing files                                                                            | `false` |
| `--diff [path]` |       | Show diffs. Without a path, shows the first 5 files. With a path, shows that file only (implies `--dry-run`)         | —       |
| `--view [path]` |       | Show file contents. Without a path, shows the first 5 files. With a path, shows that file only (implies `--dry-run`) | —       |

#### Dry-Run Mode

Use `--dry-run` to preview what `add` would do without writing any files. `--diff` and `--view` both imply `--dry-run`.

```bash
# Preview all changes.
npx shadcn@latest add button --dry-run

# Show diffs for all files (top 5).
npx shadcn@latest add button --diff

# Show the diff for a specific file.
npx shadcn@latest add button --diff button.tsx

# Show contents for all files (top 5).
npx shadcn@latest add button --view

# Show the full content of a specific file.
npx shadcn@latest add button --view button.tsx

# Works with URLs too.
npx shadcn@latest add https://api.npoint.io/abc123 --dry-run

# Works with public GitHub registries too.
npx shadcn@latest add owner/repo/item --dry-run

# CSS diffs.
npx shadcn@latest add button --diff globals.css
```

**When to use dry-run:**

- When the user asks "what files will this add?" or "what will this change?" — use `--dry-run`.
- Before overwriting existing components — use `--diff` to preview the changes first.
- When the user wants to inspect component source code without installing — use `--view`.
- When checking what CSS changes would be made to `globals.css` — use `--diff globals.css`.
- When the user asks to review or audit third-party registry code before installing — use `--view` to inspect the source.

> **`npx shadcn@latest add --dry-run` vs `npx shadcn@latest view`:** Prefer `npx shadcn@latest add --dry-run/--diff/--view` over `npx shadcn@latest view` when the user wants to preview changes to their project. `npx shadcn@latest view` only shows raw registry metadata. `npx shadcn@latest add --dry-run` shows exactly what would happen in the user's project: resolved file paths, diffs against existing files, and CSS updates. Use `npx shadcn@latest view` only when the user wants to browse registry info without a project context.

#### Smart Merge from Upstream

See [Updating Components in SKILL.md](./SKILL.md#updating-components) for the full workflow.

### `search` — Search registries

```bash
npx shadcn@latest search [registries...] [options]
```

Fuzzy search across registries. Also aliased as `npx shadcn@latest list`.
Supports namespaces (`@acme`), public GitHub registry sources (`owner/repo`),
and registry catalog URLs. Without `-q`, lists all items. When no registries are
passed, searches every registry configured in `components.json`.

| Flag                | Short | Description                                       | Default |
| ------------------- | ----- | ------------------------------------------------- | ------- |
| `--query <query>`   | `-q`  | Search query                                      | —       |
| `--type <type>`     | `-t`  | Filter by item type (e.g. `ui`, `block`, `hook`); comma-separated | —       |
| `--limit <number>`  | `-l`  | Max items to display                              | `100`   |
| `--offset <number>` | `-o`  | Items to skip                                     | `0`     |
| `--json`            |       | Output as JSON                                    | `false` |
| `--cwd <cwd>`       | `-c`  | Working directory                                 | current |

### `view` — View item details

```bash
npx shadcn@latest view <items...> [options]
```

Displays item info including file contents. Examples:
`npx shadcn@latest view @shadcn/button`,
`npx shadcn@latest view owner/repo/item`.

### `docs` — Get component documentation URLs

```bash
npx shadcn@latest docs <components...> [options]
```

Outputs resolved URLs for component documentation, examples, and API references. Accepts one or more component names. Fetch the URLs to get the actual content.

Example output for `npx shadcn@latest docs input button`:

```
base  radix

input
  docs      https://ui.shadcn.com/docs/components/radix/input
  examples  https://raw.githubusercontent.com/.../examples/input-example.tsx

button
  docs      https://ui.shadcn.com/docs/components/radix/button
  examples  https://raw.githubusercontent.com/.../examples/button-example.tsx
```

Some components include an `api` link to the underlying library (e.g. `cmdk` for the command component).

### `diff` — Check for updates

Do not use this command. Use `npx shadcn@latest add --diff` instead.

### `info` — Project information

```bash
npx shadcn@latest info [options]
```

Displays project info and `components.json` configuration. Run this first to discover the project's framework, aliases, Tailwind version, and resolved paths.

| Flag          | Short | Description       | Default |
| ------------- | ----- | ----------------- | ------- |
| `--cwd <cwd>` | `-c`  | Working directory | current |

**Project Info fields:**

| Field                | Type      | Meaning                                                            |
| -------------------- | --------- | ------------------------------------------------------------------ |
| `framework`          | `string`  | Detected framework (`next`, `vite`, `react-router`, `start`, etc.) |
| `frameworkVersion`   | `string`  | Framework version (e.g. `15.2.4`)                                  |
| `isSrcDir`           | `boolean` | Whether the project uses a `src/` directory                        |
| `isRSC`              | `boolean` | Whether React Server Components are enabled                        |
| `isTsx`              | `boolean` | Whether the project uses TypeScript                                |
| `tailwindVersion`    | `string`  | `"v3"` or `"v4"`                                                   |
| `tailwindConfigFile` | `string`  | Path to the Tailwind config file                                   |
| `tailwindCssFile`    | `string`  | Path to the global CSS file                                        |
| `aliasPrefix`        | `string`  | Import alias prefix (e.g. `@`, `~`, `@/`)                          |
| `packageManager`     | `string`  | Detected package manager (`npm`, `pnpm`, `yarn`, `bun`)            |

**Components.json fields:**

| Field                | Type      | Meaning                                                                                    |
| -------------------- | --------- | ------------------------------------------------------------------------------------------ |
| `base`               | `string`  | Primitive library (`radix` or `base`) — determines component APIs and available props      |
| `style`              | `string`  | Visual style (e.g. `nova`, `vega`)                                                         |
| `rsc`                | `boolean` | RSC flag from config                                                                       |
| `tsx`                | `boolean` | TypeScript flag                                                                            |
| `tailwind.config`    | `string`  | Tailwind config path                                                                       |
| `tailwind.css`       | `string`  | Global CSS path — this is where custom CSS variables go                                    |
| `iconLibrary`        | `string`  | Icon library — determines icon import package (e.g. `lucide-react`, `@tabler/icons-react`) |
| `aliases.components` | `string`  | Component import alias (e.g. `@/components`)                                               |
| `aliases.utils`      | `string`  | Utils import alias (e.g. `@/lib/utils`)                                                    |
| `aliases.ui`         | `string`  | UI component alias (e.g. `@/components/ui`)                                                |
| `aliases.lib`        | `string`  | Lib alias (e.g. `@/lib`)                                                                   |
| `aliases.hooks`      | `string`  | Hooks alias (e.g. `@/hooks`)                                                               |
| `resolvedPaths`      | `object`  | Absolute file-system paths for each alias                                                  |
| `registries`         | `object`  | Configured custom registries                                                               |

**Links fields:**

The `info` output includes a **Links** section with templated URLs for component docs, source, and examples. For resolved URLs, use `npx shadcn@latest docs <component>` instead.

### `build` — Build a custom registry

```bash
npx shadcn@latest build [registry] [options]
```

Builds `registry.json` into individual JSON files for distribution. Default input: `./registry.json`, default output: `./public/r`.

For authoring rules, `include`, item definitions, `registryDependencies`, and
GitHub registry behavior, see [registry.md](./registry.md).

| Flag              | Short | Description       | Default      |
| ----------------- | ----- | ----------------- | ------------ |
| `--output <path>` | `-o`  | Output directory  | `./public/r` |
| `--cwd <cwd>`     | `-c`  | Working directory | current      |

---

## Templates

| Value          | Framework      | Monorepo support |
| -------------- | -------------- | ---------------- |
| `next`         | Next.js        | Yes              |
| `vite`         | Vite           | Yes              |
| `start`        | TanStack Start | Yes              |
| `react-router` | React Router   | Yes              |
| `astro`        | Astro          | Yes              |
| `laravel`      | Laravel        | No               |

All templates support monorepo scaffolding via the `--monorepo` flag. When passed, the CLI uses a monorepo-specific template directory (e.g. `next-monorepo`, `vite-monorepo`). When neither `--monorepo` nor `--no-monorepo` is passed, the CLI prompts interactively. Laravel does not support monorepo scaffolding.

---

## Presets

Three ways to specify a preset via `--preset`:

1. **Named:** `--preset nova` or `--preset lyra`
2. **Code:** `--preset a2r6bw` (version-prefixed base62 string, e.g. `a2r6bw` or `b0`)
3. **URL:** `--preset "https://ui.shadcn.com/init?base=radix&style=nova&..."`

> **IMPORTANT:** Never try to decode, fetch, or resolve preset codes manually. Preset codes are opaque — pass them directly to `npx shadcn@latest init --preset <code>` and let the CLI handle resolution.
> Use `npx shadcn@latest apply --preset <code>` when overwriting an existing project's preset.

## Switching Presets

Ask the user first: **overwrite**, **merge**, or **skip** existing components?

- **Overwrite / Re-install** → `npx shadcn@latest apply --preset <code>`. Overwrites all detected component files with the new preset styles. Use when the user hasn't customized components.
- **Merge** → `npx shadcn@latest init --preset <code> --force --no-reinstall`, then run `npx shadcn@latest info` to get the list of installed components and use the [smart merge workflow](./SKILL.md#updating-components) to update them one by one, preserving local changes. Use when the user has customized components.
- **Skip** → `npx shadcn@latest init --preset <code> --force --no-reinstall`. Only updates config and CSS variables, leaves existing components as-is.

Always run preset commands inside the user's project directory. `apply` only works in an existing project with a `components.json` file. The CLI automatically preserves the current base (`base` vs `radix`) from `components.json`. If you must use a scratch/temp directory (e.g. for `--dry-run` comparisons), pass `--base <current-base>` explicitly — preset codes do not encode the base.
