# PM's dependency graph must remain independent of the application closure.
{
  lib,
  callPackage,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
}:
let
  python = (callPackage ./pythonLock.nix { }).interpreter;
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ../pm; };
  pythonSet = (callPackage pyproject-nix.build.packages { inherit python; }).overrideScope (
    lib.composeManyExtensions [
      pyproject-build-systems.overlays.default
      (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
    ]
  );
  environment = pythonSet.mkVirtualEnv "hermes-pm-runtime" workspace.deps.default;
in
# Nix owns this environment; no runtime download or uv resolution is needed.
environment.overrideAttrs (old: {
  postInstall = (old.postInstall or "") + ''
    printf '%s\n' '${builtins.toJSON {
      python = "${python}/bin/python3";
      sitePackages = python.sitePackages;
    }}' > "$out/pm-runtime.json"
  '';
})
