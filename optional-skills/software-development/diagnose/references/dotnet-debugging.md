# .NET Debugging & Log Playbook

Load when the bug lives in a .NET / ASP.NET Core / EF Core codebase. Feed the results back into the `diagnose` phases — most commands below are Phase 1 loop material or Phase 4 probes.

## Feedback loops (Phase 1)

| Goal | Command |
| --- | --- |
| Reproduce via test | `dotnet test --filter "FullyQualifiedName~TheFailingTest"` |
| Loop the flaky test | `dotnet test --filter ...` inside `for i in {1..50}; do ... || break; done` |
| Catch a hanging test | `dotnet test --blame-hang --blame-hang-timeout 3m` (produces a dump + Sequence.xml) |
| Catch a crashing test | `dotnet test --blame-crash` (collects a dump of the test host) |
| Reproduce via HTTP | `curl -v http://localhost:5000/path` against `dotnet run` or `dotnet watch run` |
| Verbose MSBuild/compiler info | `dotnet build -v:n` or `-bl` (binary log → inspect with MSBuild Structured Log Viewer) |

## First-response diagnostics (running process)

Order matters: cheap observation first, heavyweight artifacts last.

1. **Health snapshot** — `dotnet-counters monitor -p <pid> --refresh-interval 1`
   Watch: `CPU Usage`, `GC Heap Size`, `ThreadPool Thread Count`, `Exception Count`, `Allocation Rate`. A climbing exception count or pinned thread pool tells you which hypothesis to test first.
   One-shot without install (.NET 10 SDK): `dnx dotnet-counters monitor -p <pid>`; otherwise `dotnet tool install --global dotnet-counters`.
2. **Hang / deadlock / thread starvation** — `dotnet-stack report -p <pid>`
   Prints managed stacks for every thread. Look for many threads parked on the same lock or an async chain blocked on `.Result`/`.Wait()`.
3. **Slowness / hot path** — `dotnet-trace collect -p <pid> --profile cpu-sampling --duration 00:00:30`
   Open the `.nettrace` in PerfView / VS, or `--format speedscope` to view in https://www.speedscope.app.
4. **Crash / OOM / leak** — collect a dump, then analyze:
   ```bash
   dotnet-dump collect -p <pid> -o /tmp/app.dmp --type full
   dotnet-dump analyze /tmp/app.dmp
   ```
   Key SOS commands inside `analyze`: `clrstack` (managed stack), `clrthreads`, `pe -lines` (print exception), `dumpheap -stat` (heap histogram), `dumpasync` (async state machines stuck in flight), `analyzeoom` (last OOM info), `dso` (objects on current stack).
   Symbols missing? `dotnet-symbol /tmp/app.dmp`.
5. **Memory leak** — snapshot twice under load and diff:
   `dotnet-gcdump collect -p <pid> -o before.gcdump` … load … `after.gcdump`. Compare `dumpheap -stat` growth or open both in Visual Studio/PerfView.
6. **Production, unattended** — `dotnet-monitor` exposes `/dump`, `/trace`, `/logs`, `/metrics` endpoints and rule-based collection (e.g., collect a dump when request count spikes). Prefer it over hand-rolling dump collection in containers.

## Instrumentation (Phase 4)

- **Tag probes**: `_logger.LogWarning("[DEBUG-a4f2] state={State} input={Input}", state, input);` — grep the prefix to remove later.
- **ILogger scopes** for correlation: `using (_logger.BeginScope(new Dictionary<string,object>{["OrderId"]=id}))` — every line in the block carries `OrderId`.
- **EF Core — see the SQL**:
  ```csharp
  options.LogTo(Console.WriteLine, LogLevel.Information)
         .EnableDetailedErrors()
         .EnableSensitiveDataLogging(); // DEV ONLY — leaks parameter values
  ```
  Tag queries so the SQL grep lands on your statement: `ctx.Orders.TagWith("[DEBUG-a4f2]").Where(...)`.
  Suspect N+1: count `Executed DbCommand` lines per request in the logs.
- **HTTP calls out**: register `AddHttpClient` logging or `builder.Services.AddHttpLogging(o => ...)`; verify handler timeouts (`SocketsHttpHandler.PooledConnectionLifetime`, `Timeout`) before blaming the network.
- **Distributed context**: `Activity.Current?.Id` / W3C `traceparent` already flows through ASP.NET Core + `HttpClient` — log `Activity.Current.TraceId` so frontend↔backend logs join.
- **Structured sinks**: prefer JSON output so `jq` works: Serilog `new JsonFormatter()`, or OpenTelemetry `AddOtlpExporter`.

## Log queries

```bash
# systemd service
journalctl -u myapp.service --since "1 hour ago" -p err
journalctl -u myapp.service -f | grep '\[DEBUG-a4f2\]'

# docker / kubernetes
docker logs --since 30m --tail 500 <container>
kubectl logs deploy/myapp --since=1h --tail=-1

# JSON (Serilog/OTel console) — errors only, then drill into one trace
jq -c 'select(.Level=="Error" or .["@l"]=="Error")' app.log
jq -c 'select(.TraceId=="<traceId>")' app.log
jq -r '.["@mt"]' app.log | sort | uniq -c | sort -rn | head   # top message templates

# find the exception blocks
grep -n -A30 'System\..*Exception' app.log
```

## Common failure patterns

| Symptom | Likely cause | Check |
| --- | --- | --- |
| Request hangs forever | `.Result`/`.Wait()` deadlock, missing `await`, thread-pool starvation | `dotnet-stack report` — parked threads |
| Memory grows until OOM | rooted objects (static caches, event handlers, `HttpClient` per-request) | two `dotnet-gcdump` snapshots, `dumpheap -stat` diff |
| Intermittent slowness | GC pressure, lock contention, sync-over-async | `dotnet-counters` GC + thread pool, `dotnet-trace` contention events |
| Exception storm in logs, app "works" | swallowed catch + retry loop | `Exception Count` counter; hunt `catch {}` |
| EF query slow only in prod | missing index, N+1, parameter-sniffing | `TagWith` + `LogTo`; compare `EXPLAIN` |
| Test passes locally, fails CI | time/culture/env differences | pin `TZ`, `CultureInfo`, fake clocks; `--blame-*` |

## Docs

- https://learn.microsoft.com/dotnet/core/diagnostics/tools-overview
- https://learn.microsoft.com/dotnet/core/diagnostics/debug-memory-leak
