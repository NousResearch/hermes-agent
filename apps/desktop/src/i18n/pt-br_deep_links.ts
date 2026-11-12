import type { GatewayGroupOpenCopy, SkillDeepLinkCopy } from './types_deep_links'

export const ptBrSkillDeepLink: SkillDeepLinkCopy = {
  installTitle: (name: string) => `Instalar “${name}”?`,
  installDescription:
    'Esta habilidade estará disponível em novas sessões. Instale apenas de fontes em que você confia.',
  installTo: 'Instalar em',
  thisComputer: 'Este computador',
  installing: 'Instalando…',
  installComplete: (name: string) => `“${name}” instalado`,
  destinationChanged: 'O destino mudou. Feche esta janela e abra o link de instalação novamente.',
  installed: 'Instalado',
  source: 'Fonte'
}

export const ptBrGatewayGroupOpen: GatewayGroupOpenCopy = {
  opening: 'Abrindo conversa em grupo…',
  lookupFailed: 'Não foi possível buscar este grupo no gateway. Tente novamente.',
  missing: 'Esta conversa em grupo não existe mais.'
}
