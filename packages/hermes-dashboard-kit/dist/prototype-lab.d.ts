import type { DashboardCapabilityId, DashboardWorkspaceId } from "./contracts";
export type DashboardPrototypeVariantStatus = "draft" | "review" | "approved" | "rejected" | "promoted";
export interface DashboardPrototypeDataRequirement {
    id: string;
    label: string;
    owner: string;
    required: boolean;
    currentState: "available" | "missing" | "partial" | "unknown";
    sourceHint?: string;
}
export interface DashboardPrototypeVariant {
    id: string;
    name: string;
    status: DashboardPrototypeVariantStatus;
    workspaceFocus: DashboardWorkspaceId[];
    operatorWorkflow: string;
    referenceNotes: string[];
    dataRequirements: DashboardPrototypeDataRequirement[];
    capabilityCoverage: DashboardCapabilityId[];
    previewEvidence?: DashboardPrototypePreviewEvidence[];
    promotedComponents?: string[];
}
export interface DashboardPrototypePreviewEvidence {
    id: string;
    label: string;
    kind: "static-gallery" | "app-route" | "screenshot" | "mobbin-reference" | "notes";
    path: string;
    capturedAt?: string;
}
export interface DashboardPrototypeSet {
    id: string;
    projectId: string;
    dashboardName: string;
    createdAt: string;
    objective: string;
    operatorQuestions: string[];
    variants: DashboardPrototypeVariant[];
    selectedVariantId?: string;
    selectionRationale?: string;
    selectionEvidence?: DashboardPrototypePreviewEvidence[];
}
export interface DashboardPrototypeAssessment {
    projectId: string;
    dashboardName: string;
    variantCount: number;
    readyForReview: boolean;
    selectedVariant?: DashboardPrototypeVariant;
    missingRequirements: DashboardPrototypeDataRequirement[];
    promotionActions: string[];
}
export declare function assessDashboardPrototypeSet(prototypeSet: DashboardPrototypeSet): DashboardPrototypeAssessment;
//# sourceMappingURL=prototype-lab.d.ts.map