# .NET Test Coverage

Target: **90% line and branch** coverage. Works for **xUnit**, **NUnit**, and **MSTest** — all three run through `dotnet test`, so the coverage command is identical regardless of framework.

## Prerequisites

- `Microsoft.NET.Test.Sdk` referenced by the test project.
- A coverage collector. Two options:
  - **XPlat Code Coverage** (recommended, zero extra packages): built into `Microsoft.NET.Test.Sdk` via the Coverlet collector.
  - **Coverlet.Console** (standalone): `dotnet tool install --global coverlet.console`.

## Run coverage

### XPlat (built-in, no extra tooling)

```bash
# Collect coverage into ./TestResults
dotnet test --collect:"XPlat Code Coverage" --results-directory ./TestResults
```

Output: `TestResults/<guid>/coverage.cobertura.xml`.

### Enforce a threshold

```bash
dotnet test /p:CollectCoverage=true /p:CoverageFormat=cobertura /p:Threshold=90 /p:ThresholdType=line,branch
```

### Coverlet.Console (explicit DLL)

```bash
coverlet <TestProject>/bin/Debug/net8.0/<TestProject>.dll \
  --target "dotnet" \
  --targetargs "test --no-build" \
  --format cobertura \
  --threshold 90
```

## HTML report (reportgenerator)

```bash
# install once
dotnet tool install --global dotnet-reportgenerator-globaltool

# generate from the XPlat result
reportgenerator \
  -reports:"TestResults/**/coverage.cobertura.xml" \
  -targetdir:"CoverageReport" \
  -reporttypes:"Html;Cobertura"
```

Open `CoverageReport/index.html`.

## Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Missing `Microsoft.NET.Test.Sdk` | `--collect:"XPlat Code Coverage"` produces no XML | Add the package to the test project |
| Running coverage on the wrong config | Lower numbers than expected | Build `Release` or set `/p:Configuration=Release` |
| No threshold gate | Coverage silently regresses | Add `/p:Threshold=90` (build fails below target) |
