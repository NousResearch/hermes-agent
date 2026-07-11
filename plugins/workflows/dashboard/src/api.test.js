import { expect, it } from "vitest";
import { formatApiError } from "./api.js";

it("unwraps FastAPI detail from Error.message", () => {
  expect(formatApiError(new Error('400: {"detail":"repo_path is required"}')))
    .toBe("repo_path is required");
});

it("prefers structured field error messages", () => {
  expect(formatApiError({
    code: "workflow_input_invalid",
    message: "Repository path is required.",
  })).toBe("Repository path is required.");
});
