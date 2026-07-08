import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, screen } from "@testing-library/react";
import CronPage from "./CronPage";

// Replace the REST layer with resolved fixtures so the page renders without a
// real backend. Mirrors the vi.mock pattern used across the suite.
vi.mock("@/api/cron", () => ({
  getCronJobs: vi.fn().mockResolvedValue([
    {
      id: "job-1",
      name: "Nightly digest",
      enabled: true,
      schedule_display: "every day at 9am",
      next_run_at: "2026-07-09T09:00:00Z",
      deliver: "local",
    },
  ]),
  getCronDeliveryTargets: vi.fn().mockResolvedValue({
    targets: [{ id: "local", name: "Local", home_target_set: true, home_env_var: null }],
  }),
  pauseCronJob: vi.fn(),
  resumeCronJob: vi.fn(),
  triggerCronJob: vi.fn(),
  deleteCronJob: vi.fn(),
  createCronJob: vi.fn(),
}));

afterEach(cleanup);

describe("CronPage", () => {
  it("renders a job row with its schedule and enabled state", async () => {
    render(<CronPage />);
    // useResource fetches on mount — wait for the row to appear.
    expect(await screen.findByText("Nightly digest")).toBeTruthy();
    expect(screen.getByText("every day at 9am")).toBeTruthy();
    expect(screen.getByText("enabled")).toBeTruthy();
  });
});
