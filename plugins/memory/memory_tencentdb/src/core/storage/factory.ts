/**
 * Storage Backend Factory — creates the appropriate IStorageBackend
 * based on configuration.
 *
 * The default `local` backend is bundled with core. The optional `cos` backend
 * is loaded dynamically; when unavailable, the factory throws a clear error
 * directing the operator to either provide the backend or switch to `type=local`.
 */

import type { IStorageBackend, StorageBackendConfig, StorageLogger } from "./types.js";
import { LocalStorageBackend } from "./local-backend.js";

const TAG = "[storage][factory]";

/**
 * Create a storage backend instance based on configuration.
 *
 * Async because the optional COS backend is dynamically imported only when needed.
 *
 * @param config Backend configuration (type + backend-specific options)
 * @param logger Optional logger
 * @returns IStorageBackend instance
 */
export async function createStorageBackend(
  config: StorageBackendConfig,
  logger?: StorageLogger,
): Promise<IStorageBackend> {
  switch (config.type) {
    case "cos": {
      if (!config.credentialProvider) {
        throw new Error(`${TAG} COS backend requires a credentialProvider`);
      }

      let CosStorageBackendCtor: typeof import("../../integrations/cos/cos-backend.js").CosStorageBackend;
      try {
        ({ CosStorageBackend: CosStorageBackendCtor } = await import(
          "../../integrations/cos/cos-backend.js"
        ));
      } catch (err) {
        throw new Error(
          `${TAG} COS backend is not available in this build; ` +
            `switch to storage type=local or provide a build that includes it. ` +
            `Original error: ${err instanceof Error ? err.message : String(err)}`,
        );
      }

      logger?.info(`${TAG} Creating COS storage backend`);
      return new CosStorageBackendCtor({
        credentialProvider: config.credentialProvider,
        logger,
      });
    }

    case "local":
    default: {
      const rootDir = config.localRootDir ?? "./data/storage";
      logger?.info(`${TAG} Creating local storage backend: rootDir=${rootDir}`);
      return new LocalStorageBackend({
        rootDir,
        logger,
      });
    }
  }
}

/**
 * Create a local storage backend for development.
 * Convenience helper for quick local setup.
 */
export function createLocalStorageBackend(
  rootDir: string,
  logger?: StorageLogger,
): IStorageBackend {
  return new LocalStorageBackend({ rootDir, logger });
}
