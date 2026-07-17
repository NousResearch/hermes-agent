// First-party connector PRESET surface (epic #37 / #46 OfficeCLI) — detect-or-instruct only.
//
// OfficeCLI (github.com/iOfficeAI/OfficeCLI, Apache-2.0) already ships an MCP server
// (`officecli mcp`, stdio) — the FIRST blessed connector because it needs no cloud auth and
// demos .docx/.xlsx artifacts. We BUNDLE NOTHING: the operator installs the binary, and this
// module only (a) detects whether `officecli` is on PATH and (b) stamps out the exact
// connector config the operator drops into `boardstate.connectors.json` (the authorship
// boundary — SPEC §18 / invariant #8). The recipe + detection come straight from the broker's
// own `officeCliPreset` / `detectBinary`, so this can never emit a connector the broker rejects.

import { detectBinary, officeCliPreset, type ConnectorConfig } from "@boardstate/broker";

export type PresetSetup = {
  /** Preset id (also the default connector name). */
  id: string;
  /** Human title for a status line / setup UI. */
  title: string;
  /** Is the binary reachable on PATH right now? */
  detected: boolean;
  /** The validated connector entry to author into `boardstate.connectors.json`. */
  connector: ConnectorConfig;
  /** A human install pointer when the binary is absent; null when detected. Never auto-run. */
  install: string | null;
};

/**
 * Describe the OfficeCLI connector setup: whether `officecli` is on PATH, the exact connector
 * config to author, and (when absent) the install pointer. Reads the filesystem/PATH only —
 * spawns nothing, bundles nothing.
 */
export function officeCliSetup(): PresetSetup {
  const binary = officeCliPreset.requiresBinary;
  const detected = binary ? detectBinary(binary.command) : false;
  return {
    id: officeCliPreset.id,
    title: officeCliPreset.title,
    detected,
    connector: officeCliPreset.build(),
    install: detected ? null : (binary?.install ?? null),
  };
}

/** A one-line boot hint for the operator: is OfficeCLI ready, and the next step to enable it.
 *  HYGIENE-1: this is logged to the sidecar's stdout (→ dashboard log), so it must NOT embed
 *  connector config values (command/args) — it points at the docs instead of dumping the config. */
export function officeCliBootHint(): string {
  const setup = officeCliSetup();
  const next =
    "author boardstate.connectors.json in the state dir (see docs/connectors/officecli.md)";
  return setup.detected
    ? `[boardstate] OfficeCLI detected on PATH — enable it: ${next}.`
    : `[boardstate] OfficeCLI connector available. ${setup.install} Then ${next}.`;
}
