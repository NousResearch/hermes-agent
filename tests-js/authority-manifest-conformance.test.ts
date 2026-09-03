import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  type AdmissionRequest,
  CANONICAL_AUTHORITY_MANIFEST,
  compileAuthorityManifest,
  evaluateAuthorityOperation,
  ManifestValidationError,
} from "../apps/shared/src/authority/manifest";

const root = fileURLToPath(new URL("..", import.meta.url));

const CANONICAL_GITHUB_OPERATIONS = [
  "github.issue.metadata.write",
  "github.comment.write",
  "github.contents.write",
  "github.gitdata.write",
  "github.pull_request.create",
  "github.actions.dispatch",
] as const;

type Mutation = {
  op: "set" | "delete" | "append";
  path: string[];
  value?: unknown;
};

type MutationCase = {
  name: string;
  mutations: Mutation[];
  expected: {
    compiler_valid: boolean;
    schema_valid: boolean;
  };
};

async function readJson(path: string): Promise<unknown> {
  return JSON.parse(await readFile(new URL(path, `file://${root}/`), "utf8"));
}

function asMutableRecord(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("mutation path must traverse objects");
  }
  return value as Record<string, unknown>;
}

function applyMutations(base: unknown, mutations: Mutation[]): unknown {
  const result = structuredClone(base);
  for (const mutation of mutations) {
    let parent: unknown = result;
    for (const part of mutation.path.slice(0, -1)) {
      parent = asMutableRecord(parent)[part];
    }
    const key = mutation.path.at(-1);
    if (key === undefined) {
      throw new TypeError("mutation path must not be empty");
    }
    const record = asMutableRecord(parent);
    if (mutation.op === "set") {
      record[key] = structuredClone(mutation.value);
    } else if (mutation.op === "delete") {
      delete record[key];
    } else {
      const target = record[key];
      if (!Array.isArray(target)) {
        throw new TypeError("append mutation target must be an array");
      }
      target.push(structuredClone(mutation.value));
    }
  }
  return result;
}

describe("authority manifest cross-language conformance", () => {
  it("packages the exact canonical bytes, hash, and immutable compiler output", async () => {
    const sourceBytes = await readFile(new URL("authority/manifest.v1.json", `file://${root}/`));
    const artifact = CANONICAL_AUTHORITY_MANIFEST;
    const runtimeBytes = Buffer.from(artifact.manifest_bytes);

    expect(runtimeBytes).toEqual(sourceBytes);
    expect(createHash("sha256").update(runtimeBytes).digest("hex")).toBe(
      artifact.manifest_sha256,
    );
    expect(artifact.manifest.policyVersion).toBe("2026.08.25-v2");
    expect(Object.isFrozen(artifact)).toBe(true);
    expect(Object.isFrozen(artifact.manifest_bytes)).toBe(true);
    expect(Object.isFrozen(artifact.manifest)).toBe(true);
    expect(() => {
      (artifact.manifest_bytes as number[]).push(0);
    }).toThrow(TypeError);
  });

  it("pins the six canonical GitHub operation classes and one-to-one sinks", () => {
    const manifest = CANONICAL_AUTHORITY_MANIFEST.manifest;
    const domain = manifest.domains["github.operation"];

    expect(Object.keys(domain.operations).sort()).toEqual(
      [...CANONICAL_GITHUB_OPERATIONS].sort(),
    );
    expect(Object.keys(domain.sinks).sort()).toEqual([...CANONICAL_GITHUB_OPERATIONS].sort());

    for (const [operationName, operation] of Object.entries(domain.operations)) {
      expect(operation.sink_class).toBe(operationName);
    }

    expect(domain.operations["github.contents.write"].required_capabilities).toEqual([
      "contents:write",
    ]);
    expect(domain.operations["github.gitdata.write"].required_capabilities).toEqual([
      "git_objects:write",
      "refs:write",
    ]);
    expect(domain.operations["github.pull_request.create"].required_capabilities).toEqual([
      "pull_requests:create",
    ]);
    expect(domain.operations["github.actions.dispatch"].required_capabilities).toEqual([
      "actions:dispatch",
    ]);
  });

  it("runs the canonical vectors through the TypeScript evaluator", async () => {
    const manifest = CANONICAL_AUTHORITY_MANIFEST.manifest;

    const vectors = (await readJson("authority/conformance.v1.json")) as {
      schema_version: number;
      vectors: Array<{
        name: string;
        request: AdmissionRequest;
        expected: unknown;
      }>;
    };

    expect(vectors.schema_version).toBe(1);

    for (const vector of vectors.vectors) {
      expect(evaluateAuthorityOperation(manifest, vector.request), vector.name).toEqual(
        vector.expected,
      );
    }
  });

  it("runs the shared accepted/rejected mutation corpus through the compiler", async () => {
    const canonical = await readJson("authority/manifest.v1.json");
    const corpus = (await readJson("authority/manifest-mutations.v1.json")) as {
      schema_version: number;
      cases: MutationCase[];
    };

    expect(corpus.schema_version).toBe(1);
    for (const testCase of corpus.cases) {
      const candidate = applyMutations(canonical, testCase.mutations);
      if (testCase.expected.compiler_valid) {
        expect(() => compileAuthorityManifest(candidate), testCase.name).not.toThrow();
      } else {
        expect(() => compileAuthorityManifest(candidate), testCase.name).toThrow(
          ManifestValidationError,
        );
      }
    }
  });

  it("retains immutable sink broker and direct-symbol ownership", () => {
    const manifest = CANONICAL_AUTHORITY_MANIFEST.manifest;
    const domain = manifest.domains["github.operation"];
    const sink = domain.sinks["github.comment.write"];

    expect(sink).toEqual({
      broker: "githubMutationBroker",
      direct_symbols: ["issues.createComment", "pulls.createReview", "pulls.createReviewComment"],
    });
    for (const operation of Object.values(domain.operations)) {
      expect(domain.sinks[operation.sink_class]).toBeDefined();
    }
  });

  it("does not expose runtime-mutable compiled policy", () => {
    const manifest = CANONICAL_AUTHORITY_MANIFEST.manifest;
    const request: AdmissionRequest = {
      domain: "github.operation",
      operation_class: "github.comment.write",
      actor_class: "human",
      resource_state: "open",
      capabilities: [
        {
          capability: "comments:write",
          granted: true,
          source: "user_api_token",
          generation: "cred-immutability",
        },
      ],
    };
    const before = evaluateAuthorityOperation(manifest, request);

    expect(() => {
      (manifest.domains as Record<string, unknown>)["ambient.operation"] = {};
    }).toThrow(TypeError);
    expect(() => {
      const required = manifest.domains["github.operation"].operations["github.comment.write"]
        .required_capabilities as string[];
      required.push("admin:*");
    }).toThrow(TypeError);
    expect(() => {
      const sink = manifest.domains["github.operation"].sinks[
        "github.comment.write"
      ] as { broker: string };
      sink.broker = "ambientBroker";
    }).toThrow(TypeError);

    expect(evaluateAuthorityOperation(manifest, request)).toEqual(before);
  });
});
