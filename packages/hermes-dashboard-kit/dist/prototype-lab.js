export function assessDashboardPrototypeSet(prototypeSet) {
    const selectedVariant = prototypeSet.selectedVariantId
        ? prototypeSet.variants.find((variant) => variant.id === prototypeSet.selectedVariantId)
        : undefined;
    const missingRequirements = prototypeSet.variants.flatMap((variant) => (variant.dataRequirements.filter((requirement) => requirement.required && requirement.currentState !== "available")));
    const promotionActions = [];
    if (prototypeSet.variants.length < 3) {
        promotionActions.push("Create at least three comparable dashboard variants before selecting a direction.");
    }
    if (!prototypeSet.operatorQuestions.length) {
        promotionActions.push("Define the operator questions before visual review.");
    }
    if (!selectedVariant) {
        promotionActions.push("Select a variant and record the selection rationale.");
    }
    if (selectedVariant && !prototypeSet.selectionRationale) {
        promotionActions.push("Record why the selected variant best supports the operator workflow.");
    }
    if (selectedVariant && !prototypeSet.selectionEvidence?.length && !selectedVariant.previewEvidence?.length) {
        promotionActions.push("Attach preview evidence before promoting the selected variant.");
    }
    if (selectedVariant?.status === "approved" && !selectedVariant.promotedComponents?.length) {
        promotionActions.push("List the dashboard-kit components or local adapters that should be promoted.");
    }
    if (missingRequirements.length) {
        promotionActions.push("Resolve required data requirements before treating the prototype as production-ready.");
    }
    return {
        projectId: prototypeSet.projectId,
        dashboardName: prototypeSet.dashboardName,
        variantCount: prototypeSet.variants.length,
        readyForReview: prototypeSet.variants.length >= 3 && prototypeSet.operatorQuestions.length > 0,
        selectedVariant,
        missingRequirements,
        promotionActions,
    };
}
//# sourceMappingURL=prototype-lab.js.map