import type { GatewayHostTranslations } from './types_gateway_host'

export const ptBrGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Configurações do gateway indisponíveis',
  unavailableDesc:
    'As configurações de conexão só podem ser alteradas pelo aplicativo Hermes Desktop no computador em que ele é executado.',
  webappHostTitle: 'Computador do Hermes',
  webappHostDesc:
    'O aplicativo web sempre é executado no computador do Hermes que o disponibiliza. As conexões salvas abaixo são a lista deste computador, compartilhada com o aplicativo Hermes Desktop; outros gateways são acessados por meio deste computador.',
  webappOauthPaste: {
    title: 'Concluir login',
    description:
      'Se a aba de login terminou em um endereço que não carregou (este navegador não está no computador do gateway), cole esse endereço aqui.',
    placeholder: 'http://127.0.0.1:…/webapp/gateway-connections/oauth/callback?…',
    finish: 'Concluir',
    failed: 'Não foi possível concluir o login'
  },
  webAccess: {
    title: 'Acesso pela web',
    intro:
      'Acesse o Hermes deste computador pelo celular ou por outro computador, entrando com sua conta Nous. Quem entrar poderá executar comandos neste computador.',
    failed: 'Não foi possível alterar o acesso pela web',
    running: 'Em execução',
    stopped: 'Parado',
    starting: 'Iniciando',
    stopping: 'Parando',
    restartNeeded: 'É necessário reiniciar',
    unavailable: 'Indisponível',
    notRunningDetail: 'Inicie para acessar o Hermes de outro dispositivo.',
    needsNous: 'Primeiro, entre na sua conta Nous em Provedores.',
    runningLan: 'Abra este endereço em um dispositivo na mesma rede.',
    runningPublic: 'Disponível pela sua URL.',
    runningLocal:
      'Somente este computador pode acessar. Pare e inicie novamente para permitir acesso de outros dispositivos.',
    pendingOn: 'Salvo. O gateway disponibilizará o aplicativo web após reiniciar.',
    pendingOff: 'Desativado, mas o gateway continua disponibilizando o aplicativo web até reiniciar.',
    unavailableDetail: reason => `O gateway não conseguiu disponibilizar o aplicativo web: ${reason}`,
    restartBusy:
      'Salvo. Há uma conversa em andamento, por isso o gateway não reiniciou. Escolha Reiniciar gateway quando ela terminar.',
    restartFailed: 'Salvo, mas o gateway não conseguiu reiniciar. Escolha Reiniciar gateway para tentar novamente.',
    restartFailedBecause: reason => `O gateway não conseguiu reiniciar: ${reason}`,
    startTimedOut: 'O gateway não disponibilizou o aplicativo web em 90 segundos.',
    stopTimedOut: 'O gateway não parou de disponibilizar o aplicativo web em 90 segundos.',
    startStages: {
      preparing: 'Preparando o aplicativo web. A compilação pode levar alguns minutos.',
      building: 'Compilando o aplicativo web. Isso pode levar alguns minutos.',
      registering: 'Configurando o login da Nous para este endereço.',
      restarting: 'Reiniciando o gateway para disponibilizar o aplicativo web.'
    },
    startElapsed: time => `${time} decorridos`,
    start: 'Iniciar',
    stop: 'Parar',
    restartGateway: 'Reiniciar gateway',
    modeLan: 'Esta rede',
    modeLanDetail: url => `Dispositivos no seu Wi-Fi acessam ${url}. Use apenas em uma rede em que você confia.`,
    lanUnencrypted: 'A conexão não é criptografada: outras pessoas nesta rede podem ler o que você envia e recebe.',
    noLan: 'Este computador não tem um endereço de rede acessível por outros dispositivos.',
    modePublic: 'Minha própria URL',
    modePublicDetail: port =>
      `Um endereço HTTPS que você já usa, como Tailscale Serve ou um túnel, encaminhando para http://127.0.0.1:${port}.`
  }
}
