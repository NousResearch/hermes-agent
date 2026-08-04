/**
 * Offload MMD Query Handler — returns MMD files.
 * Body params: sessionId (required), limit (optional, default=all)
 * When limit=1, only returns the active MMD.
 */
import type http from "node:http";
import type { StorageAdapter } from "../core/storage/adapter.js";
import type { OffloadState } from "./types.js";
import { createHash } from "node:crypto";
import { buildOffloadBasePath } from "./session-utils.js";

function hashContent(content: string): string {
  return createHash("md5").update(content).digest("hex").slice(0, 12);
}

export async function handleMmdQuery(
  _req: http.IncomingMessage,
  res: http.ServerResponse,
  _auth: { serviceId: string },
  storage: StorageAdapter,
  requestId: string,
  sendJson: (res: http.ServerResponse, status: number, body: unknown) => void,
  successEnvelope: <T>(data: T, requestId: string) => unknown,
  errorEnvelope: (code: number, message: string, requestId: string) => unknown,
  sessionId: string,
  limit?: number,
): Promise<void> {
  if (!sessionId) {
    sendJson(res, 400, errorEnvelope(400, "missing sessionId", requestId));
    return;
  }

  const basePath = buildOffloadBasePath(sessionId);

  // Read state
  const stateRaw = await storage.readFile(`${basePath}/state.json`);
  let state: Partial<OffloadState> = {};
  if (stateRaw) {
    try {
      state = JSON.parse(stateRaw);
    } catch {
      // ignore
    }
  }

  const mmdsPrefix = `${basePath}/mmds/`;
  const mmds: Array<{
    filename: string;
    content: string;
    version: string;
  }> = [];

  if (limit === 1 && state.activeMmdFile) {
    // Fast path: only return the active MMD
    const content = await storage.readFile(`${mmdsPrefix}${state.activeMmdFile}`) ?? "";
    mmds.push({
      filename: state.activeMmdFile,
      content,
      version: content ? hashContent(content) : "",
    });
  } else {
    // Return all (or up to limit)
    const mmdFiles = await storage.readdirNames(mmdsPrefix, ".mmd");
    const filesToRead = limit && limit > 0 ? mmdFiles.slice(0, limit) : mmdFiles;

    const readResults = await Promise.all(
      filesToRead.map(async (filename) => {
        const content = await storage.readFile(`${mmdsPrefix}${filename}`) ?? "";
        return {
          filename,
          content,
          version: content ? hashContent(content) : "",
        };
      }),
    );
    mmds.push(...readResults);
  }

  sendJson(
    res,
    200,
    successEnvelope(
      {
        mmds,
        currentMmd: state.activeMmdFile ?? null,
      },
      requestId,
    ),
  );
}
