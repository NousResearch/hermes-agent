/**
 * Core abstractions — host/edition-neutral interfaces.
 *
 * This barrel file aggregates all dependency-inversion interfaces that decouple
 * core business logic (memory recall, scene, L1/L2/L3 pipeline, gateway HTTP
 * layer) from edition-specific adapter implementations (default vs enhanced).
 *
 * Design intent:
 *   Core code under src/core/ and src/gateway/ should depend ONLY on the
 *   interfaces re-exported here. Concrete implementations can be provided by
 *   the default local modules or by optional deployment-specific adapters.
 *
 * The gateway chooses a dependency set at startup based on deployMode. The
 * default local implementation should remain fully buildable by itself.
 */

export type {
  IConfigSource,
  IQuotaReporter,
  QuotaSnapshot,
} from "./types.js";
