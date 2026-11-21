export function physicalSidebarControls<T>(edges: { left: T; right: T }, rtl: boolean) {
  return {
    left: rtl ? edges.right : edges.left,
    right: rtl ? edges.left : edges.right,
    leftActionId: rtl ? 'view.toggleRightSidebar' : 'view.toggleSidebar',
    rightActionId: rtl ? 'view.toggleSidebar' : 'view.toggleRightSidebar',
    leftTour: rtl ? 'right-pane-toggle' : undefined,
    rightTour: rtl ? undefined : 'right-pane-toggle'
  } as const
}
