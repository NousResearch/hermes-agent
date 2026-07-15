import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApprovalsSection } from "./ApprovalsSection";
import { type ApprovalPreview } from "../mockData";

const approval: ApprovalPreview = {
  id: "opaque-a",
  title: "Опасная операция",
  source: "gateway",
  risk: "critical",
  summary: "Нужна ручная проверка",
  requestedAt: "сейчас",
  status: "ожидает",
  checks: ["owner", "risk"],
  allowedDecisions: ["approve_once", "reject_once"],
};

function renderSection(overrides: Partial<Parameters<typeof ApprovalsSection>[0]> = {}) {
  const props: Parameters<typeof ApprovalsSection>[0] = {
    approvals: [approval],
    selectedId: approval.id,
    onSelect: vi.fn(),
    actionsEnabled: true,
    canSubmitDecision: true,
    decisionBlockReason: "",
    decisionError: "",
    isOwner: true,
    isConnected: true,
    onDecision: vi.fn(async () => true),
    ...overrides,
  };
  return { ...render(<ApprovalsSection {...props} />), props };
}

describe("ApprovalsSection decision readiness", () => {
  it("does not open confirmation when only actionsEnabled is true but snapshot_version readiness is missing", () => {
    renderSection({
      canSubmitDecision: false,
      decisionBlockReason: "Нет свежей версии очереди одобрений. Обнови снимок данных.",
    });

    expect(screen.getByText("Нет свежей версии очереди одобрений. Обнови снимок данных.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Одобрить позже" })).toBeDisabled();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("keeps the confirmation sheet open and shows visible error when decision POST fails", async () => {
    const onDecision = vi.fn(async () => false);
    const { rerender, props } = renderSection({ onDecision });

    fireEvent.click(screen.getByRole("button", { name: "Одобрить" }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    rerender(<ApprovalsSection {...props} decisionError="Очередь одобрений устарела. Я обновил снимок — проверь запрос и попробуй снова." />);
    fireEvent.click(screen.getByRole("button", { name: "Одобрить один раз" }));

    await waitFor(() => expect(onDecision).toHaveBeenCalledWith(approval.id, "approve_once"));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getAllByRole("alert").some((node) => node.textContent?.includes("Очередь одобрений устарела"))).toBe(true);
  });

  it("exposes an external close callback for Telegram BackButton to dismiss the confirmation sheet", async () => {
    let closeConfirm: (() => void) | undefined;
    const onConfirmOpenChange = vi.fn((isOpen: boolean, close?: () => void) => {
      closeConfirm = close;
    });

    renderSection({ onConfirmOpenChange });
    fireEvent.click(screen.getByRole("button", { name: "Одобрить" }));

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await waitFor(() => expect(onConfirmOpenChange).toHaveBeenCalledWith(true, expect.any(Function)));

    act(() => {
      closeConfirm?.();
    });

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(onConfirmOpenChange).toHaveBeenLastCalledWith(false);
  });

  it("uses record-only copy and never says gateway already applies decisions", () => {
    renderSection();

    expect(screen.getAllByText(/record-only/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/подключены/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/исполняет gateway/i)).not.toBeInTheDocument();
  });
});
