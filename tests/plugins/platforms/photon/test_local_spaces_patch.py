"""Regression tests for the temporary Spectrum local-iMessage spaces patch."""
from __future__ import annotations

import subprocess
from pathlib import Path


_PATCHER = Path("plugins/platforms/photon/sidecar/patch-spectrum-local-spaces.mjs")


def test_sidecar_applies_local_spaces_patch_before_sdk_import() -> None:
    script = r'''
import { createSpectrumRuntime } from "./plugins/platforms/photon/sidecar/spectrum-runtime.mjs";
const calls = [];
const importer = async (specifier) => {
  calls.push(`import:${specifier}`);
  if (specifier === "@spectrum-ts/core") {
    return {
      Spectrum: async () => ({}),
      attachment: null,
      voice: null,
      text: null,
      markdown: null,
      richlink: null,
      typing: null,
      poll: null,
    };
  }
  return { localIMessage: { config: () => ({}) } };
};
await createSpectrumRuntime({
  localMode: true,
  projectId: null,
  projectSecret: null,
  telemetry: false,
  importer,
  patchLocalSpaces: () => calls.push("patch"),
});
if (calls[0] !== "patch") throw new Error(JSON.stringify(calls));
'''
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_patch_allows_local_space_get_and_is_idempotent(tmp_path: Path) -> None:
    dist = tmp_path / "node_modules" / "spectrum-ts" / "dist"
    dist.mkdir(parents=True)
    chunk = dist / "chunk-imessage.js"
    chunk.write_text(
        """var provider = {
    get: async ({ input, client }) => {
      if (isLocal(client)) {
        throw UnsupportedError.action(
          "space.get",
          "iMessage (local mode)",
          "local mode only supports replying to existing messages"
        );
      }
      return { id: input.id };
    }
};
""",
        encoding="utf-8",
    )

    first = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    second = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    patched = chunk.read_text(encoding="utf-8")
    assert "Hermes local iMessage workaround: allow space.get" in patched
    assert "id: input.id" in patched
    assert 'phone: "local"' in patched
    assert "UnsupportedError.action" not in patched
