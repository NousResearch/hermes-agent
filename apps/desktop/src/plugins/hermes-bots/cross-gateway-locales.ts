/** Creating group chats whose bots may run on other computers: every new one is run by a gateway. Every Bot Mode locale. */
export interface CrossGatewayGroupMessages {
  gatewayPeer: (owner: string, local?: boolean) => string
  pickBots: (max: number) => string
  gatewayMembership: string
  checkingPeers: string
  peerCheckFailed: (connection: string, local?: boolean) => string
  /** The owner's Hermes cannot run group chats at all. */
  ownerUpdateRequired: (place: string) => string
  /** The owner's gateway cannot seat bots from other computers. */
  ownerCannotSeatPeers: (place: string) => string
  /** A selected bot's own computer is not connected to this Desktop. */
  peerRouteMissing: string
  /** A bot's connection now reaches another Hermes installation than the bot came from. */
  peerGatewayChanged: (place: string) => string
  /** A bot's machine is offline: Manage groups cannot change its rooms. */
  groupsOffline: (connection: string, bot: string) => string
  /** A room member's machine is offline: Manage members cannot remove it. */
  memberOffline: (connection: string, bot: string) => string
  /** Save stopped because a member's machine did not take the change. */
  memberSaveFailed: (connection: string, bot: string) => string
  /** Save stopped, and putting back a change already made on a member's machine did not land. */
  memberRollbackFailed: (connection: string, bot: string) => string
  selectionElsewhere: (connection: string) => string
}

export const CROSS_GATEWAY_GROUP_LOCALES = {
  en: {
    gatewayPeer: (owner, local = false) => `Runs on ${local ? 'this device' : owner}, including when this Desktop is closed. Only text is sent to agents on other computers.`,
    pickBots: max => `Pick 2–${max} agents.`,
    gatewayMembership: 'Membership is saved with this group chat on its computer.',
    checkingPeers: 'Checking that every computer in this group chat is ready…',
    peerCheckFailed: (connection, local = false) => `Could not check ${local ? 'this device' : connection}. Make sure Hermes is running there, then retry.`,
    ownerUpdateRequired: place => `Hermes on ${place} is too old to run group chats. Update Hermes there, then retry.`,
    ownerCannotSeatPeers: place => `Hermes on ${place} can’t seat agents from other computers yet. Update Hermes there and run its gateway (hermes gateway run), then retry.`,
    peerRouteMissing: 'A selected agent’s computer isn’t connected. Reconnect it, then retry.',
    peerGatewayChanged: place => `${place} now runs a different Hermes installation than this agent came from. Reconnect it, then retry.`,
    groupsOffline: (connection, bot) => `${connection} is offline — reconnect to change ${bot}’s groups.`,
    memberOffline: (connection, bot) => `${connection} is offline — reconnect to remove ${bot}.`,
    memberSaveFailed: (connection, bot) => `Members not saved: ${connection} did not take the change for ${bot}. Nothing was changed; try again once it is reachable.`,
    memberRollbackFailed: (connection, bot) =>
      `Members not saved, and the change already made for ${bot} on ${connection} could not be undone: ${connection} may still have it. The group chat shows what ${connection} last reported; reconnect it and check the members.`,
    selectionElsewhere: connection => `These agents all live on ${connection}. Add an agent from this connection, or switch to ${connection} to start a group of its agents.`
  },
  ja: {
    gatewayPeer: (owner, local = false) => `このデスクトップを閉じていても ${local ? 'このデバイス' : owner} で動作します。ほかのコンピューターのエージェントはテキストのみを受け取ります。`,
    pickBots: max => `2〜${max} 個のエージェントを選択してください。`,
    gatewayMembership: 'メンバー情報は、このグループチャットを実行するコンピューターに保存されます。',
    checkingPeers: 'このグループチャットのすべてのコンピューターの準備ができているか確認しています…',
    peerCheckFailed: (connection, local = false) => `${local ? 'このデバイス' : connection} を確認できませんでした。そこで Hermes が動作していることを確認してから、再試行してください。`,
    ownerUpdateRequired: place => `${place} の Hermes が古いため、グループチャットを実行できません。そこで Hermes を更新してから、再試行してください。`,
    ownerCannotSeatPeers: place => `${place} の Hermes は、まだほかのコンピューターのエージェントを参加させられません。そこで Hermes を更新し、ゲートウェイ（hermes gateway run）を実行してから、再試行してください。`,
    peerRouteMissing: '選択したエージェントのコンピューターが接続されていません。再接続してから、再試行してください。',
    peerGatewayChanged: place => `${place} は、このエージェントとは別の Hermes インストールで動作しています。再接続してから、再試行してください。`,
    groupsOffline: (connection, bot) => `${connection} はオフラインです。${bot} のグループを変更するには再接続してください。`,
    memberOffline: (connection, bot) => `${connection} はオフラインです。${bot} を外すには再接続してください。`,
    memberSaveFailed: (connection, bot) => `メンバーを保存できませんでした: ${connection} が ${bot} の変更を受け付けませんでした。何も変更されていません。接続できるようになってから再試行してください。`,
    memberRollbackFailed: (connection, bot) =>
      `メンバーを保存できませんでした。${connection} 上の ${bot} にすでに加えた変更を元に戻せませんでした。${connection} には変更が残っている可能性があります。グループチャットには ${connection} が最後に報告した状態を表示しています。再接続してメンバーを確認してください。`,
    selectionElsewhere: connection => `これらのエージェントはすべて ${connection} にあります。この接続のエージェントを追加するか、${connection} に切り替えてそのエージェントだけのグループを作成してください。`
  },
  zh: {
    gatewayPeer: (owner, local = false) => `在 ${local ? '此设备' : owner} 上运行，即使此桌面端已关闭。其他电脑上的智能体仅接收文本。`,
    pickBots: max => `选择 2–${max} 个智能体。`,
    gatewayMembership: '成员关系随此群聊保存在运行它的电脑上。',
    checkingPeers: '正在检查此群聊中的每台电脑是否已就绪…',
    peerCheckFailed: (connection, local = false) => `无法检查 ${local ? '此设备' : connection}。请确认那里的 Hermes 正在运行，然后重试。`,
    ownerUpdateRequired: place => `${place} 上的 Hermes 版本太旧，无法运行群聊。请在那里更新 Hermes，然后重试。`,
    ownerCannotSeatPeers: place => `${place} 上的 Hermes 暂时无法让其他电脑上的智能体加入。请在那里更新 Hermes 并运行其网关（hermes gateway run），然后重试。`,
    peerRouteMissing: '所选智能体的电脑未连接。请重新连接，然后重试。',
    peerGatewayChanged: place => `${place} 现在运行的 Hermes 安装与此智能体来源的不同。请重新连接，然后重试。`,
    groupsOffline: (connection, bot) => `${connection} 已离线——重新连接后才能更改 ${bot} 的群组。`,
    memberOffline: (connection, bot) => `${connection} 已离线——重新连接后才能移除 ${bot}。`,
    memberSaveFailed: (connection, bot) => `成员未保存：${connection} 未接受 ${bot} 的更改。未做任何更改；请在可连接后重试。`,
    memberRollbackFailed: (connection, bot) =>
      `成员未保存，且已对 ${connection} 上的 ${bot} 做出的更改无法撤销：${connection} 可能仍保留该更改。群聊显示的是 ${connection} 最后报告的状态；请重新连接并检查成员。`,
    selectionElsewhere: connection => `这些智能体都在 ${connection} 上。请添加一个此连接上的智能体，或切换到 ${connection} 来创建只包含其智能体的群组。`
  },
  'zh-hant': {
    gatewayPeer: (owner, local = false) => `在 ${local ? '此裝置' : owner} 上執行，即使此桌面端已關閉。其他電腦上的智慧體僅接收文字。`,
    pickBots: max => `選擇 2–${max} 個智慧體。`,
    gatewayMembership: '成員關係隨此群組聊天儲存在執行它的電腦上。',
    checkingPeers: '正在檢查此群組聊天中的每台電腦是否已就緒…',
    peerCheckFailed: (connection, local = false) => `無法檢查 ${local ? '此裝置' : connection}。請確認那裡的 Hermes 正在執行，然後再試一次。`,
    ownerUpdateRequired: place => `${place} 上的 Hermes 版本太舊，無法執行群組聊天。請在那裡更新 Hermes，然後再試一次。`,
    ownerCannotSeatPeers: place => `${place} 上的 Hermes 暫時無法讓其他電腦上的智慧體加入。請在那裡更新 Hermes 並執行其閘道（hermes gateway run），然後再試一次。`,
    peerRouteMissing: '所選智慧體的電腦未連線。請重新連線，然後再試一次。',
    peerGatewayChanged: place => `${place} 現在執行的 Hermes 安裝與此智慧體來源的不同。請重新連線，然後再試一次。`,
    groupsOffline: (connection, bot) => `${connection} 已離線——重新連線後才能變更 ${bot} 的群組。`,
    memberOffline: (connection, bot) => `${connection} 已離線——重新連線後才能移除 ${bot}。`,
    memberSaveFailed: (connection, bot) => `成員未儲存：${connection} 未接受 ${bot} 的變更。未做任何變更；請在可連線後重試。`,
    memberRollbackFailed: (connection, bot) =>
      `成員未儲存，且已對 ${connection} 上的 ${bot} 做出的變更無法復原：${connection} 可能仍保留該變更。群組聊天顯示的是 ${connection} 最後回報的狀態；請重新連線並檢查成員。`,
    selectionElsewhere: connection => `這些智慧體都在 ${connection} 上。請加入一個此連線上的智慧體，或切換到 ${connection} 來建立只包含其智慧體的群組。`
  },
  ar: {
    gatewayPeer: (owner, local = false) => `تعمل على ${local ? 'هذا الجهاز' : owner} حتى عند إغلاق سطح المكتب هذا. تقبل الوكلاء على الأجهزة الأخرى النص فقط.`,
    pickBots: max => `اختر من 2 إلى ${max} بوتات.`,
    gatewayMembership: 'تُحفظ العضوية مع محادثة المجموعة هذه على الجهاز الذي يشغّلها.',
    checkingPeers: 'جارٍ التحقق من جاهزية كل جهاز في محادثة المجموعة هذه…',
    peerCheckFailed: (connection, local = false) => `تعذّر التحقق من ${local ? 'هذا الجهاز' : connection}. تأكد من أن Hermes يعمل هناك، ثم أعد المحاولة.`,
    ownerUpdateRequired: place => `إصدار Hermes على ${place} قديم جدًا لتشغيل محادثات المجموعة. حدّث Hermes هناك، ثم أعد المحاولة.`,
    ownerCannotSeatPeers: place => `لا يستطيع Hermes على ${place} بعدُ ضمّ وكلاء من أجهزة أخرى. حدّث Hermes هناك وشغّل بوابته (hermes gateway run)، ثم أعد المحاولة.`,
    peerRouteMissing: 'جهاز أحد الوكلاء المحددة غير متصل. أعد الاتصال به، ثم أعد المحاولة.',
    peerGatewayChanged: place => `يعمل ${place} الآن بتثبيت Hermes مختلف عن الذي جاء منه هذا الوكيل. أعد الاتصال به، ثم أعد المحاولة.`,
    groupsOffline: (connection, bot) => `${connection} غير متصل — أعد الاتصال لتغيير مجموعات ${bot}.`,
    memberOffline: (connection, bot) => `${connection} غير متصل — أعد الاتصال لإزالة ${bot}.`,
    memberSaveFailed: (connection, bot) => `لم تُحفظ قائمة الأعضاء: لم يقبل ${connection} التغيير الخاص بـ ${bot}. لم يتغيّر شيء؛ أعد المحاولة عندما يصبح متاحًا.`,
    memberRollbackFailed: (connection, bot) =>
      `لم تُحفظ قائمة الأعضاء، وتعذّر التراجع عن التغيير الذي أُجري بالفعل لـ ${bot} على ${connection}: قد يظل التغيير موجودًا على ${connection}. تعرض محادثة المجموعة آخر ما أبلغ عنه ${connection}؛ أعد الاتصال به وتحقق من الأعضاء.`,
    selectionElsewhere: connection => `كل هذه الوكلاء موجودة على ${connection}. أضف وكيلًا من هذا الاتصال، أو انتقل إلى ${connection} لبدء مجموعة من وكلاءه.`
  },
  ru: {
    gatewayPeer: (owner, local = false) => `Работает на ${local ? 'этом устройстве' : owner}, даже когда этот Desktop закрыт. На других компьютерах агенты принимают только текст.`,
    pickBots: max => `Выберите от 2 до ${max} агентов.`,
    gatewayMembership: 'Участники сохраняются вместе с групповым чатом на компьютере, где он работает.',
    checkingPeers: 'Проверяем, готов ли каждый компьютер этого группового чата…',
    peerCheckFailed: (connection, local = false) => `Не удалось проверить ${local ? 'это устройство' : connection}. Убедитесь, что там запущен Hermes, и повторите.`,
    ownerUpdateRequired: place => `Hermes на ${place} слишком старый для групповых чатов. Обновите там Hermes и повторите.`,
    ownerCannotSeatPeers: place => `Hermes на ${place} пока не может принимать агентов с других компьютеров. Обновите там Hermes, запустите его шлюз (hermes gateway run) и повторите.`,
    peerRouteMissing: 'Компьютер одного из выбранных агентов не подключён. Переподключите его и повторите.',
    peerGatewayChanged: place => `На ${place} теперь работает другая установка Hermes, не та, из которой этот агент. Переподключите его и повторите.`,
    groupsOffline: (connection, bot) => `${connection} не в сети — переподключитесь, чтобы изменить группы ${bot}.`,
    memberOffline: (connection, bot) => `${connection} не в сети — переподключитесь, чтобы убрать ${bot}.`,
    memberSaveFailed: (connection, bot) => `Участники не сохранены: ${connection} не принял изменение для ${bot}. Ничего не изменено; повторите, когда он станет доступен.`,
    memberRollbackFailed: (connection, bot) =>
      `Участники не сохранены, и уже внесённое изменение для ${bot} на ${connection} не удалось отменить: на ${connection} оно может остаться. Групповой чат показывает последнее, что сообщил ${connection}; переподключитесь и проверьте участников.`,
    selectionElsewhere: connection => `Все эти агенты находятся на ${connection}. Добавьте агента из этого подключения или переключитесь на ${connection}, чтобы создать группу из его агентов.`
  },
  fr: {
    gatewayPeer: (owner, local = false) => `Fonctionne sur ${local ? 'cet appareil' : owner}, même lorsque ce Desktop est fermé. Les agents des autres ordinateurs acceptent uniquement du texte.`,
    pickBots: max => `Choisissez de 2 à ${max} agents.`,
    gatewayMembership: 'Les membres sont enregistrés avec cette discussion de groupe sur son ordinateur.',
    checkingPeers: 'Vérification que chaque ordinateur de cette discussion de groupe est prêt…',
    peerCheckFailed: (connection, local = false) => `Impossible de vérifier ${local ? 'cet appareil' : connection}. Vérifiez que Hermes y fonctionne, puis réessayez.`,
    ownerUpdateRequired: place => `Hermes sur ${place} est trop ancien pour les discussions de groupe. Mettez-y Hermes à jour, puis réessayez.`,
    ownerCannotSeatPeers: place => `Hermes sur ${place} ne peut pas encore accueillir de agents d’autres ordinateurs. Mettez-y Hermes à jour et lancez sa passerelle (hermes gateway run), puis réessayez.`,
    peerRouteMissing: 'L’ordinateur d’un agent sélectionné n’est pas connecté. Reconnectez-le, puis réessayez.',
    peerGatewayChanged: place => `${place} exécute maintenant une autre installation de Hermes que celle de cet agent. Reconnectez-le, puis réessayez.`,
    groupsOffline: (connection, bot) => `${connection} est hors ligne — reconnectez-le pour modifier les groupes de ${bot}.`,
    memberOffline: (connection, bot) => `${connection} est hors ligne — reconnectez-le pour retirer ${bot}.`,
    memberSaveFailed: (connection, bot) => `Membres non enregistrés : ${connection} n’a pas pris la modification pour ${bot}. Rien n’a changé ; réessayez quand il sera joignable.`,
    memberRollbackFailed: (connection, bot) =>
      `Membres non enregistrés, et la modification déjà faite pour ${bot} sur ${connection} n’a pas pu être annulée : ${connection} l’a peut-être encore. La discussion de groupe affiche le dernier état signalé par ${connection} ; reconnectez-le et vérifiez les membres.`,
    selectionElsewhere: connection => `Ces agents se trouvent tous sur ${connection}. Ajoutez un agent de cette connexion, ou passez sur ${connection} pour créer un groupe avec ses agents.`
  },
  de: {
    gatewayPeer: (owner, local = false) => `Läuft auf ${local ? 'diesem Gerät' : owner}, auch wenn dieser Desktop geschlossen ist. Auf anderen Computern empfangen Agenten nur Text.`,
    pickBots: max => `Wählen Sie 2–${max} Agenten.`,
    gatewayMembership: 'Die Mitglieder werden mit diesem Gruppenchat auf seinem Computer gespeichert.',
    checkingPeers: 'Es wird geprüft, ob jeder Computer dieses Gruppenchats bereit ist…',
    peerCheckFailed: (connection, local = false) => `${local ? 'Dieses Gerät' : connection} konnte nicht geprüft werden. Prüfen Sie, ob Hermes dort läuft, und versuchen Sie es erneut.`,
    ownerUpdateRequired: place => `Hermes auf ${place} ist zu alt für Gruppenchats. Aktualisieren Sie Hermes dort und versuchen Sie es erneut.`,
    ownerCannotSeatPeers: place => `Hermes auf ${place} kann noch keine Agenten anderer Computer aufnehmen. Aktualisieren Sie Hermes dort, starten Sie sein Gateway (hermes gateway run) und versuchen Sie es erneut.`,
    peerRouteMissing: 'Der Computer eines ausgewählten Agenten ist nicht verbunden. Verbinden Sie ihn neu und versuchen Sie es erneut.',
    peerGatewayChanged: place => `Auf ${place} läuft jetzt eine andere Hermes-Installation als die, von der dieser Agent stammt. Verbinden Sie ihn neu und versuchen Sie es erneut.`,
    groupsOffline: (connection, bot) => `${connection} ist offline – verbinde es neu, um die Gruppen von ${bot} zu ändern.`,
    memberOffline: (connection, bot) => `${connection} ist offline – verbinde es neu, um ${bot} zu entfernen.`,
    memberSaveFailed: (connection, bot) => `Mitglieder nicht gespeichert: ${connection} hat die Änderung für ${bot} nicht übernommen. Nichts wurde geändert; versuche es erneut, sobald es erreichbar ist.`,
    memberRollbackFailed: (connection, bot) =>
      `Mitglieder nicht gespeichert, und die bereits vorgenommene Änderung für ${bot} auf ${connection} ließ sich nicht rückgängig machen: ${connection} hat sie möglicherweise noch. Der Gruppenchat zeigt den zuletzt von ${connection} gemeldeten Stand; verbinde es neu und prüfe die Mitglieder.`,
    selectionElsewhere: connection => `Diese Agenten liegen alle auf ${connection}. Füge einen Agenten dieser Verbindung hinzu oder wechsle zu ${connection}, um eine Gruppe mit dessen Agenten zu starten.`
  },
  es: {
    gatewayPeer: (owner, local = false) => `Funciona en ${local ? 'este dispositivo' : owner}, incluso cuando este Desktop está cerrado. Los agentes de otros equipos solo reciben texto.`,
    pickBots: max => `Elige de 2 a ${max} agentes.`,
    gatewayMembership: 'Los miembros se guardan con este chat de grupo en el equipo que lo ejecuta.',
    checkingPeers: 'Comprobando que cada equipo de este chat de grupo está listo…',
    peerCheckFailed: (connection, local = false) => `No se pudo comprobar ${local ? 'este dispositivo' : connection}. Comprueba que Hermes se esté ejecutando allí y vuelve a intentarlo.`,
    ownerUpdateRequired: place => `Hermes en ${place} es demasiado antiguo para los chats de grupo. Actualiza Hermes allí y vuelve a intentarlo.`,
    ownerCannotSeatPeers: place => `Hermes en ${place} todavía no puede incorporar agentes de otros equipos. Actualiza Hermes allí, ejecuta su pasarela (hermes gateway run) y vuelve a intentarlo.`,
    peerRouteMissing: 'El equipo de un agente seleccionado no está conectado. Vuelve a conectarlo y vuelve a intentarlo.',
    peerGatewayChanged: place => `${place} ejecuta ahora otra instalación de Hermes distinta de la de este agente. Vuelve a conectarlo y vuelve a intentarlo.`,
    groupsOffline: (connection, bot) => `${connection} está sin conexión: vuelve a conectarlo para cambiar los grupos de ${bot}.`,
    memberOffline: (connection, bot) => `${connection} está sin conexión: vuelve a conectarlo para quitar a ${bot}.`,
    memberSaveFailed: (connection, bot) => `Miembros no guardados: ${connection} no aceptó el cambio de ${bot}. No se cambió nada; inténtalo de nuevo cuando esté disponible.`,
    memberRollbackFailed: (connection, bot) =>
      `Miembros no guardados, y el cambio ya hecho para ${bot} en ${connection} no se pudo deshacer: puede que ${connection} todavía lo tenga. El chat de grupo muestra lo último que informó ${connection}; vuelve a conectarlo y revisa los miembros.`,
    selectionElsewhere: connection => `Todos estos agentes están en ${connection}. Añade un agente de esta conexión o cambia a ${connection} para crear un grupo con sus agentes.`
  }
} satisfies Record<string, CrossGatewayGroupMessages>
