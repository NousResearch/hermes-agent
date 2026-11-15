export async function fetchRosterSourceData<T>(
  fetchProfiles: () => Promise<T>,
  fetchInstallId: () => Promise<string | undefined>
): Promise<{ body: T; installId: string | undefined }> {
  const [body, installId] = await Promise.all([fetchProfiles(), fetchInstallId()])

  return { body, installId }
}
