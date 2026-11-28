import type { MemoryLayersTranslations } from './types_memory_layers'

// Settings > Memory & Context and Command Center > Maintenance.
export const svMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'Aktiv minnesleverantör',
  selectedProvider: 'Vald minnesleverantör',
  selectedDesc: current => `Inte aktiv ännu: sessioner använder fortfarande ${current} tills valet har sparats.`,
  builtinMemory: 'endast inbyggt minne',
  noProvider: 'Ingen – endast inbyggt minne',
  noProviderDesc:
    'Hermes använder bara det inbyggda lagret nedan. Välj en extern leverantör under Inställningar › Minne och kontext för att lägga till en.',
  providerDesc: name => `${name} lägger till sin egen minnesåterkallning i varje session, utöver det inbyggda lagret.`,
  providerNotReadyDesc: name =>
    `${name} är den sparade leverantören men lägger inte till någon minnesåterkallning förrän den är klar. Det inbyggda lagret fungerar ändå.`,
  providerUnknownDesc: name => `${name} är den sparade leverantören; Hermes kunde inte läsa om den är klar.`,
  statusUnavailable: 'Minnesstatusen är inte tillgänglig',
  statusUnavailableDesc: 'Kunde inte läsa vilken minnesleverantör som är aktiv eller om den är klar.',
  retry: 'Försök igen',
  status: {
    ready: 'klar',
    needs_config: 'måste ställas in',
    unavailable: 'inte tillgänglig',
    missing: 'inte installerad'
  },
  builtinLayer: 'Inbyggt lager',
  onByDefault: 'På som standard',
  on: 'På',
  off: 'Av',
  onlyFile: file => `endast ${file}`,
  builtinDesc:
    'MEMORY.md och USER.md är Hermes eget minne och är skilda från externa leverantörer. Båda läses in i sessioner som standard. Växlarna för minne och användarprofil under Inställningar › Minne och kontext stänger av dem var för sig. En leverantör lägger till detta lager, men ersätter det aldrig.'
}
