export async function createSpectrumRuntime({
  localMode,
  projectId,
  projectSecret,
  telemetry,
  importer = (specifier) => import(specifier),
}) {
  const core = await importer("@spectrum-ts/core");
  const provider = await importer(
    localMode
      ? "@spectrum-ts/imessage-local"
      : "spectrum-ts/providers/imessage",
  );
  const imessage = localMode ? provider.localIMessage : provider.imessage;

  const app = await core.Spectrum({
    ...(localMode ? {} : { projectId, projectSecret }),
    providers: [imessage.config()],
    options: { flattenGroups: true },
    telemetry,
  });

  return {
    app,
    imessage,
    attachment: core.attachment,
    voice: core.voice,
    text: core.text,
    markdown: core.markdown,
    richlink: core.richlink,
    typing: core.typing,
    poll: core.poll,
    effect: provider.effect,
  };
}
