/**
 * resolveSkillConfig — pure function: user-facing SkillConfigInput + ambient capabilities
 * (SkillEnvProbe) → resolved ResolvedSkillConfig + downgrade list.
 *
 * Design: SKILL_ENGINEERING_DESIGN.md §11.1.4
 * - Single-step downgrade per field, no recursive probing.
 * - Exactly one log line per downgrade at the right level.
 * - skill.enabled=false or missing → return null, log nothing.
 * - Resolution is deterministic given input + probe; no env reads.
 *
 * Defaults (when input.enabled=true and field is unspecified):
 * - storeBackend: inherit probe.outerStoreBackend, fallback "sqlite"
 * - contentBackend: auto-probe: cos if hasCosCredentials else local
 * - routing.mode: "bm25"
 * - routing.hybridAlpha: 0.3
 * - routing.searchTopK: 20
 * - routing.charBudgetPercent: 0.01
 * - routing.fastPathMinNameLength: 4
 * - extraction.enabled: false
 * - extraction.maxIterations: 16
 * - resources.maxResourceSizeBytes: 5_000_000
 * - resources.allowExecutable: false
 * - resources.downloadDir: "/tmp/tdai-skill-resources"
 */

import { SkillCoreError } from "./skill-core.js";
import type {
  ResolvedSkillConfig,
  SkillConfigInput,
  SkillDegradation,
  SkillEnvProbe,
} from "./types.js";

export interface ResolverLogger {
  debug?: (msg: string) => void;
  info: (msg: string) => void;
  warn: (msg: string) => void;
  error: (msg: string) => void;
}

const TAG = "[skill][config]";

/**
 * @param strictMode — when true, COS/contentBackend degradations throw instead of silently
 *   falling back to local. Use in service mode where COS is always required.
 */
export function resolveSkillConfig(
  input: SkillConfigInput | undefined,
  probe: SkillEnvProbe,
  logger: ResolverLogger,
  strictMode = false,
): ResolvedSkillConfig | null {
  if (!input?.enabled) {
    return null;
  }

  const degradations: SkillDegradation[] = [];

  // --------------- store ---------------
  const requestedStore =
    input.storeBackend ?? probe.outerStoreBackend ?? "sqlite";
  let storeBackend: "sqlite" | "tcvdb" = requestedStore;
  if (storeBackend === "tcvdb" && !probe.hasTcvdbCredentials) {
    degradations.push({
      field: "storeBackend",
      from: "tcvdb",
      to: "sqlite",
      reason: "TCVDB credentials missing (url / apiKey / database)",
      level: "warn",
    });
    logger.warn(
      `${TAG} storeBackend=tcvdb requested but credentials missing — degrading to sqlite`,
    );
    storeBackend = "sqlite";
  }

  // --------------- content ---------------
  const explicitContent = input.contentBackend;
  let contentBackend: "local" | "cos";
  if (explicitContent === "cos") {
    if (probe.hasCosCredentials) {
      contentBackend = "cos";
    } else if (strictMode) {
      throw new SkillCoreError(
        "SKILL_COS_REQUIRED",
        "contentBackend=cos required but COS credentials missing (secretId / secretKey / bucket). " +
          "Verify Shark COS config or cos.env in service mode — refusing to silently fall back to local fs.",
      );
    } else {
      degradations.push({
        field: "contentBackend",
        from: "cos",
        to: "local",
        reason: "COS credentials missing (secretId / secretKey / bucket)",
        level: "info",
      });
      logger.info(
        `${TAG} contentBackend=cos requested but credentials missing — degrading to local fs`,
      );
      contentBackend = "local";
    }
  } else if (explicitContent === "local") {
    contentBackend = "local";
  } else {
    // auto-probe: prefer COS when available
    contentBackend = probe.hasCosCredentials ? "cos" : "local";
  }

  // --------------- routing ---------------
  const requestedMode = input.routing?.mode ?? "bm25";
  let routingMode: "bm25" | "embedding" | "hybrid" = requestedMode;
  if (
    (routingMode === "embedding" || routingMode === "hybrid") &&
    !probe.embeddingAvailable
  ) {
    degradations.push({
      field: "routing.mode",
      from: routingMode,
      to: "bm25",
      reason: "embedding subsystem unavailable (disabled / provider invalid)",
      level: "warn",
    });
    logger.warn(
      `${TAG} routing.mode=${routingMode} requested but embedding unavailable — degrading to bm25`,
    );
    routingMode = "bm25";
  }

  // --------------- extraction ---------------
  const extractionEnabled = input.extraction?.enabled === true;
  if (extractionEnabled && !probe.llmRunnerAvailable) {
    // /v3/skill/extract will return empty candidates when LLMRunner is absent
    degradations.push({
      field: "extraction.runtime",
      from: "enabled",
      to: "noop",
      reason: "no LLMRunnerFactory provided by host",
      level: "warn",
    });
    logger.warn(
      `${TAG} extraction.enabled=true but no LLMRunner — extract will return empty candidates`,
    );
  }

  // --------------- archiveBytes (单一归档尺寸旋钮) ---------------
  // 派生 7 个内部字段: bytesThreshold / requestCompressThresholdBytes /
  // chunkMaxBytes / headKeepBytes / tailKeepBytes / headChars / tailChars。
  // 无效值 (<=0 或非整数) 落回默认 40KB + warn。
  const DEFAULT_ARCHIVE_BYTES = 40 * 1024;
  const rawArchive = input.extraction?.archiveBytes;
  let archiveBytes = DEFAULT_ARCHIVE_BYTES;
  if (rawArchive !== undefined) {
    if (!Number.isInteger(rawArchive) || rawArchive <= 0) {
      logger.warn(
        `${TAG} extraction.archiveBytes=${rawArchive} invalid (must be positive integer); ` +
          `falling back to default ${DEFAULT_ARCHIVE_BYTES}`,
      );
    } else {
      archiveBytes = rawArchive;
    }
  }

  const resolved: ResolvedSkillConfig = {
    enabled: true,
    storeBackend,
    contentBackend,
    routing: {
      mode: routingMode,
      hybridAlpha: input.routing?.hybridAlpha ?? 0.3,
      searchTopK: input.routing?.searchTopK ?? 20,
      charBudgetPercent: input.routing?.charBudgetPercent ?? 0.01,
      fastPathMinNameLength: input.routing?.fastPathMinNameLength ?? 4,
    },
    extraction: {
      enabled: extractionEnabled,
      toolCallThreshold: input.extraction?.toolCallThreshold ?? 10,
      model: input.extraction?.model,
      maxIterations: input.extraction?.maxIterations ?? 16,
      archiveBytes,
      maxTokens: input.extraction?.maxTokens,
      // 派生字段 — 命名和默认值来自 add-handler.ts / oversize-strategy.ts
      // 的 DEFAULT_* 常量，保持"改一个 archiveBytes 一切随动"的语义。
      bytesThreshold: archiveBytes,
      requestCompressThresholdBytes: archiveBytes,
      chunkMaxBytes: 2 * archiveBytes,
      headKeepBytes: archiveBytes,
      tailKeepBytes: archiveBytes,
      headChars: archiveBytes,
      tailChars: archiveBytes,
    },
    compress: {
      toolContentThresholdBytes: input.compress?.toolContentThresholdBytes ?? 2048,
      headBytes: input.compress?.headBytes ?? 1024,
      tailBytes: input.compress?.tailBytes ?? 1024,
    },
    resources: {
      maxResourceSizeBytes:
        input.resources?.maxResourceSizeBytes ?? 5_000_000,
      downloadDir: input.resources?.downloadDir ?? "/tmp/tdai-skill-resources",
      allowExecutable: input.resources?.allowExecutable === true,
    },
    versionTtlSeconds: (input.versionTtlDays ?? 0) * 86400,
    degradations,
  };

  logger.info(
    `${TAG} initialized: storeBackend=${resolved.storeBackend}, ` +
      `contentBackend=${resolved.contentBackend}, ` +
      `routing.mode=${resolved.routing.mode}, ` +
      `extraction=${resolved.extraction.enabled}, ` +
      `degradations=${resolved.degradations.length}`,
  );

  return resolved;
}