/** Keep test launches from changing the user's OS-level protocol handler. */
export function registerDeepLinkProtocolOutsideTests(
  testWorkerIndex: string | undefined,
  register: () => void
): void {
  if (testWorkerIndex !== undefined) {
    return
  }

  register()
}
