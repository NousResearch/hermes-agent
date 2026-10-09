import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into fr.ts.
export const frGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Paramètres du gateway indisponibles',
  unavailableDesc: "Le pont IPC du desktop n'expose pas les paramètres du gateway.",
  webappHostTitle: 'Hôte Hermes',
  webappHostDesc:
    "La Webapp utilise toujours l'hôte Hermes qui la sert. Pour changer de gateway, vous connecter à Hermes Cloud ou gérer les connexions enregistrées, utilisez l'application Hermes Desktop."
}
