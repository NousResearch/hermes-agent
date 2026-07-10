import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, cleanup, screen, fireEvent, waitFor } from "@testing-library/react";
import * as skillsApi from "@/api/skills";
import SkillsPage from "./SkillsPage";

// Mock the whole api module so the page renders without any real network I/O.
vi.mock("@/api/skills", () => ({
  getSkills: vi.fn(),
  getToolsets: vi.fn(),
  toggleSkill: vi.fn(),
  toggleToolset: vi.fn(),
  searchSkillsHub: vi.fn(),
  installSkillFromHub: vi.fn(),
}));

afterEach(cleanup);

describe("SkillsPage toggle logic", () => {
  beforeEach(() => {
    vi.mocked(skillsApi.getSkills).mockResolvedValue([
      { name: "alpha", description: "First skill", category: "core", enabled: true },
      { name: "beta", description: "Second skill", category: "core", enabled: false },
    ]);
    vi.mocked(skillsApi.getToolsets).mockResolvedValue([
      {
        name: "web",
        label: "Web",
        description: "browse the web",
        enabled: true,
        configured: true,
        tools: ["fetch", "search"],
      },
    ]);
    vi.mocked(skillsApi.toggleSkill).mockResolvedValue({ ok: true });
    vi.mocked(skillsApi.toggleToolset).mockResolvedValue({ ok: true, name: "web", enabled: false });
  });

  it("renders installed skills and toolsets", async () => {
    render(<SkillsPage />);
    expect(await screen.findByText("alpha")).toBeTruthy();
    expect(screen.getByText("beta")).toBeTruthy();
    expect(screen.getByText("Web")).toBeTruthy();
  });

  it("disabling an enabled skill sends enabled=false", async () => {
    render(<SkillsPage />);
    await screen.findByText("alpha");
    fireEvent.click(screen.getByLabelText("Disable alpha"));
    await waitFor(() => expect(skillsApi.toggleSkill).toHaveBeenCalledWith("alpha", false));
  });

  it("enabling a disabled skill sends enabled=true", async () => {
    render(<SkillsPage />);
    await screen.findByText("beta");
    fireEvent.click(screen.getByLabelText("Enable beta"));
    await waitFor(() => expect(skillsApi.toggleSkill).toHaveBeenCalledWith("beta", true));
  });

  it("toggling an enabled toolset sends enabled=false", async () => {
    render(<SkillsPage />);
    await screen.findByText("Web");
    fireEvent.click(screen.getByLabelText("Disable Web"));
    await waitFor(() => expect(skillsApi.toggleToolset).toHaveBeenCalledWith("web", false));
  });
});
