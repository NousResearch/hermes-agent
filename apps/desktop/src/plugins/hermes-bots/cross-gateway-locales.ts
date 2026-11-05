/** Execution lifetime for groups spanning connected machines. Every Bot Mode locale. */
export interface CrossGatewayGroupMessages {
  gatewayPeer: (owner: string, local?: boolean) => string
  startGatewayGroup: string
  upgradeHistory: string
  pickBots: (max: number) => string
  desktopMembership: string
  gatewayMembership: string
  checkingPeers: string
  peerCheckFailed: (connection: string, local?: boolean) => string
  desktopDriven: string
  /** Why a selection with a named profile on another computer keeps Desktop in the loop. */
  namedProfileFallback: string
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
    gatewayPeer: (owner, local = false) => `Runs on ${local ? 'this device' : owner}, including when this Desktop is closed. Only text is sent to bots on other computers.`,
    startGatewayGroup: 'Start a group chat that runs without Desktop',
    upgradeHistory: 'Earlier messages stay in this group chat.',
    pickBots: max => `Pick 2–${max} bots.`,
    desktopMembership: 'Local memberships sync through each bot profile; cross-machine members stay scoped to this group chat.',
    gatewayMembership: 'Membership is saved with this group chat on its computer.',
    checkingPeers: 'Checking whether these computers can run the group chat without Desktop…',
    peerCheckFailed: (connection, local = false) => `Could not check ${local ? 'this device' : connection}. Retry to find out whether this group chat can run without Desktop.`,
    desktopDriven:
      'Runs while this Desktop is open: bots on other machines answer from their own gateways, and this Desktop carries the messages between them.',
    namedProfileFallback: 'Bots from a named profile on another computer can’t join a group chat that runs without Desktop yet.',
    groupsOffline: (connection, bot) => `${connection} is offline — reconnect to change ${bot}’s groups.`,
    memberOffline: (connection, bot) => `${connection} is offline — reconnect to remove ${bot}.`,
    memberSaveFailed: (connection, bot) => `Members not saved: ${connection} did not take the change for ${bot}. Nothing was changed; try again once it is reachable.`,
    memberRollbackFailed: (connection, bot) =>
      `Members not saved, and the change already made for ${bot} on ${connection} could not be undone: ${connection} may still have it. The group chat shows what ${connection} last reported; reconnect it and check the members.`,
    selectionElsewhere: connection => `These bots all live on ${connection}. Add a bot from this connection, or switch to ${connection} to start a group of its bots.`
  },
  ja: {
    gatewayPeer: (owner, local = false) => `このデスクトップを閉じていても ${local ? 'このデバイス' : owner} で動作します。ほかのコンピューターのボットはテキストのみを受け取ります。`,
    startGatewayGroup: 'Desktop なしで動くグループチャットを始める',
    upgradeHistory: '過去のメッセージはこのグループチャットに残ります。',
    pickBots: max => `2〜${max} 個のボットを選択してください。`,
    desktopMembership: 'ローカルのメンバー情報は各ボットのプロファイルを通じて同期されます。別のコンピューターのメンバーはこのグループチャットにのみ保存されます。',
    gatewayMembership: 'メンバー情報は、このグループチャットを実行するコンピューターに保存されます。',
    checkingPeers: 'デスクトップなしでグループチャットを実行できるか確認しています…',
    peerCheckFailed: (connection, local = false) => `${local ? 'このデバイス' : connection} を確認できませんでした。再試行して、このグループチャットを Desktop なしで実行できるか確認してください。`,
    desktopDriven:
      'このデスクトップを開いている間だけ動作します。ほかのマシンのボットはそれぞれのゲートウェイから返信し、このデスクトップがメッセージを中継します。',
    namedProfileFallback: '別のコンピューターの名前付きプロファイルのボットは、デスクトップなしで動作するグループチャットにはまだ参加できません。',
    groupsOffline: (connection, bot) => `${connection} はオフラインです。${bot} のグループを変更するには再接続してください。`,
    memberOffline: (connection, bot) => `${connection} はオフラインです。${bot} を外すには再接続してください。`,
    memberSaveFailed: (connection, bot) => `メンバーを保存できませんでした: ${connection} が ${bot} の変更を受け付けませんでした。何も変更されていません。接続できるようになってから再試行してください。`,
    memberRollbackFailed: (connection, bot) =>
      `メンバーを保存できませんでした。${connection} 上の ${bot} にすでに加えた変更を元に戻せませんでした。${connection} には変更が残っている可能性があります。グループチャットには ${connection} が最後に報告した状態を表示しています。再接続してメンバーを確認してください。`,
    selectionElsewhere: connection => `これらのボットはすべて ${connection} にあります。この接続のボットを追加するか、${connection} に切り替えてそのボットだけのグループを作成してください。`
  },
  zh: {
    gatewayPeer: (owner, local = false) => `在 ${local ? '此设备' : owner} 上运行，即使此桌面端已关闭。其他电脑上的机器人仅接收文本。`,
    startGatewayGroup: '新建无需 Desktop 也能运行的群聊',
    upgradeHistory: '之前的消息保留在此群聊中。',
    pickBots: max => `选择 2–${max} 个机器人。`,
    desktopMembership: '本地成员关系通过各机器人的配置文件同步；跨电脑成员仅属于此群聊。',
    gatewayMembership: '成员关系随此群聊保存在运行它的电脑上。',
    checkingPeers: '正在检查这些电脑能否在桌面端关闭后运行群聊…',
    peerCheckFailed: (connection, local = false) => `无法检查 ${local ? '此设备' : connection}。请重试，确认此群聊能否在 Desktop 关闭后运行。`,
    desktopDriven: '仅在此桌面端打开时运行：其他机器上的机器人通过各自的网关回复，由此桌面端在它们之间传递消息。',
    namedProfileFallback: '其他电脑上命名配置文件的机器人暂时无法加入无需桌面端运行的群聊。',
    groupsOffline: (connection, bot) => `${connection} 已离线——重新连接后才能更改 ${bot} 的群组。`,
    memberOffline: (connection, bot) => `${connection} 已离线——重新连接后才能移除 ${bot}。`,
    memberSaveFailed: (connection, bot) => `成员未保存：${connection} 未接受 ${bot} 的更改。未做任何更改；请在可连接后重试。`,
    memberRollbackFailed: (connection, bot) =>
      `成员未保存，且已对 ${connection} 上的 ${bot} 做出的更改无法撤销：${connection} 可能仍保留该更改。群聊显示的是 ${connection} 最后报告的状态；请重新连接并检查成员。`,
    selectionElsewhere: connection => `这些机器人都在 ${connection} 上。请添加一个此连接上的机器人，或切换到 ${connection} 来创建只包含其机器人的群组。`
  },
  'zh-hant': {
    gatewayPeer: (owner, local = false) => `在 ${local ? '此裝置' : owner} 上執行，即使此桌面端已關閉。其他電腦上的機器人僅接收文字。`,
    startGatewayGroup: '建立無需 Desktop 也能執行的群聊',
    upgradeHistory: '之前的訊息保留在此群聊中。',
    pickBots: max => `選擇 2–${max} 個機器人。`,
    desktopMembership: '本機成員關係透過各機器人的設定檔同步；跨電腦成員僅屬於此群組聊天。',
    gatewayMembership: '成員關係隨此群組聊天儲存在執行它的電腦上。',
    checkingPeers: '正在檢查這些電腦能否在桌面端關閉後執行群組聊天…',
    peerCheckFailed: (connection, local = false) => `無法檢查 ${local ? '此裝置' : connection}。請重試，確認此群組聊天能否在 Desktop 關閉後執行。`,
    desktopDriven: '僅在此桌面端開啟時執行：其他機器上的機器人透過各自的閘道回覆，由此桌面端在它們之間傳遞訊息。',
    namedProfileFallback: '其他電腦上具名設定檔的機器人暫時無法加入無需桌面端執行的群組聊天。',
    groupsOffline: (connection, bot) => `${connection} 已離線——重新連線後才能變更 ${bot} 的群組。`,
    memberOffline: (connection, bot) => `${connection} 已離線——重新連線後才能移除 ${bot}。`,
    memberSaveFailed: (connection, bot) => `成員未儲存：${connection} 未接受 ${bot} 的變更。未做任何變更；請在可連線後重試。`,
    memberRollbackFailed: (connection, bot) =>
      `成員未儲存，且已對 ${connection} 上的 ${bot} 做出的變更無法復原：${connection} 可能仍保留該變更。群組聊天顯示的是 ${connection} 最後回報的狀態；請重新連線並檢查成員。`,
    selectionElsewhere: connection => `這些機器人都在 ${connection} 上。請加入一個此連線上的機器人，或切換到 ${connection} 來建立只包含其機器人的群組。`
  },
  ar: {
    gatewayPeer: (owner, local = false) => `تعمل على ${local ? 'هذا الجهاز' : owner} حتى عند إغلاق سطح المكتب هذا. تقبل الروبوتات على الأجهزة الأخرى النص فقط.`,
    startGatewayGroup: 'بدء محادثة جماعية تعمل دون Desktop',
    upgradeHistory: 'تبقى الرسائل السابقة في هذه المحادثة الجماعية.',
    pickBots: max => `اختر من 2 إلى ${max} بوتات.`,
    desktopMembership: 'تتم مزامنة العضوية المحلية عبر ملف تعريف كل بوت؛ وتبقى عضوية الأجهزة الأخرى خاصة بهذه المحادثة الجماعية.',
    gatewayMembership: 'تُحفظ العضوية مع هذه المحادثة الجماعية على الجهاز الذي يشغّلها.',
    checkingPeers: 'جارٍ التحقق مما إذا كانت هذه الأجهزة تستطيع تشغيل المحادثة الجماعية دون سطح المكتب…',
    peerCheckFailed: (connection, local = false) => `تعذّر التحقق من ${local ? 'هذا الجهاز' : connection}. أعد المحاولة لمعرفة ما إذا كانت المحادثة الجماعية تعمل دون Desktop.`,
    desktopDriven:
      'تعمل ما دام سطح المكتب هذا مفتوحًا: تردّ الروبوتات على الأجهزة الأخرى من بواباتها الخاصة، وينقل سطح المكتب هذا الرسائل بينها.',
    namedProfileFallback: 'لا يمكن بعد للروبوتات من ملف تعريف مسمّى على جهاز كمبيوتر آخر الانضمام إلى محادثة مجموعة تعمل من دون سطح المكتب.',
    groupsOffline: (connection, bot) => `${connection} غير متصل — أعد الاتصال لتغيير مجموعات ${bot}.`,
    memberOffline: (connection, bot) => `${connection} غير متصل — أعد الاتصال لإزالة ${bot}.`,
    memberSaveFailed: (connection, bot) => `لم تُحفظ قائمة الأعضاء: لم يقبل ${connection} التغيير الخاص بـ ${bot}. لم يتغيّر شيء؛ أعد المحاولة عندما يصبح متاحًا.`,
    memberRollbackFailed: (connection, bot) =>
      `لم تُحفظ قائمة الأعضاء، وتعذّر التراجع عن التغيير الذي أُجري بالفعل لـ ${bot} على ${connection}: قد يظل التغيير موجودًا على ${connection}. تعرض محادثة المجموعة آخر ما أبلغ عنه ${connection}؛ أعد الاتصال به وتحقق من الأعضاء.`,
    selectionElsewhere: connection => `كل هذه الروبوتات موجودة على ${connection}. أضف روبوتًا من هذا الاتصال، أو انتقل إلى ${connection} لبدء مجموعة من روبوتاته.`
  },
  ru: {
    gatewayPeer: (owner, local = false) => `Работает на ${local ? 'этом устройстве' : owner}, даже когда этот Desktop закрыт. На других компьютерах боты принимают только текст.`,
    startGatewayGroup: 'Начать групповой чат, работающий без Desktop',
    upgradeHistory: 'Предыдущие сообщения останутся в этом групповом чате.',
    pickBots: max => `Выберите от 2 до ${max} ботов.`,
    desktopMembership: 'Локальное участие синхронизируется через профиль каждого бота; участие ботов с других компьютеров сохраняется только в этом групповом чате.',
    gatewayMembership: 'Участники сохраняются вместе с групповым чатом на компьютере, где он работает.',
    checkingPeers: 'Проверяем, могут ли эти компьютеры запускать групповой чат без Desktop…',
    peerCheckFailed: (connection, local = false) => `Не удалось проверить ${local ? 'это устройство' : connection}. Повторите проверку, чтобы узнать, может ли групповой чат работать без Desktop.`,
    desktopDriven:
      'Работает, пока открыт этот Desktop: боты на других машинах отвечают через свои шлюзы, а этот Desktop передаёт сообщения между ними.',
    namedProfileFallback: 'Боты из именованного профиля на другом компьютере пока не могут присоединиться к групповому чату, который работает без Desktop.',
    groupsOffline: (connection, bot) => `${connection} не в сети — переподключитесь, чтобы изменить группы ${bot}.`,
    memberOffline: (connection, bot) => `${connection} не в сети — переподключитесь, чтобы убрать ${bot}.`,
    memberSaveFailed: (connection, bot) => `Участники не сохранены: ${connection} не принял изменение для ${bot}. Ничего не изменено; повторите, когда он станет доступен.`,
    memberRollbackFailed: (connection, bot) =>
      `Участники не сохранены, и уже внесённое изменение для ${bot} на ${connection} не удалось отменить: на ${connection} оно может остаться. Групповой чат показывает последнее, что сообщил ${connection}; переподключитесь и проверьте участников.`,
    selectionElsewhere: connection => `Все эти боты находятся на ${connection}. Добавьте бота из этого подключения или переключитесь на ${connection}, чтобы создать группу из его ботов.`
  },
  fr: {
    gatewayPeer: (owner, local = false) => `Fonctionne sur ${local ? 'cet appareil' : owner}, même lorsque ce Desktop est fermé. Les bots des autres ordinateurs acceptent uniquement du texte.`,
    startGatewayGroup: 'Créer une discussion de groupe qui fonctionne sans Desktop',
    upgradeHistory: 'Les anciens messages restent dans cette discussion de groupe.',
    pickBots: max => `Choisissez de 2 à ${max} bots.`,
    desktopMembership: 'Les membres locaux sont synchronisés via le profil de chaque bot ; les membres des autres ordinateurs restent propres à cette discussion de groupe.',
    gatewayMembership: 'Les membres sont enregistrés avec cette discussion de groupe sur son ordinateur.',
    checkingPeers: 'Vérification de la capacité de ces ordinateurs à faire fonctionner la discussion de groupe sans Desktop…',
    peerCheckFailed: (connection, local = false) => `Impossible de vérifier ${local ? 'cet appareil' : connection}. Réessayez pour savoir si cette discussion de groupe peut fonctionner sans Desktop.`,
    desktopDriven:
      'Fonctionne tant que ce Desktop est ouvert : les bots des autres machines répondent depuis leur propre passerelle, et ce Desktop transmet les messages entre eux.',
    namedProfileFallback: 'Les bots d’un profil nommé sur un autre ordinateur ne peuvent pas encore rejoindre une discussion de groupe qui fonctionne sans Desktop.',
    groupsOffline: (connection, bot) => `${connection} est hors ligne — reconnectez-le pour modifier les groupes de ${bot}.`,
    memberOffline: (connection, bot) => `${connection} est hors ligne — reconnectez-le pour retirer ${bot}.`,
    memberSaveFailed: (connection, bot) => `Membres non enregistrés : ${connection} n’a pas pris la modification pour ${bot}. Rien n’a changé ; réessayez quand il sera joignable.`,
    memberRollbackFailed: (connection, bot) =>
      `Membres non enregistrés, et la modification déjà faite pour ${bot} sur ${connection} n’a pas pu être annulée : ${connection} l’a peut-être encore. La discussion de groupe affiche le dernier état signalé par ${connection} ; reconnectez-le et vérifiez les membres.`,
    selectionElsewhere: connection => `Ces bots se trouvent tous sur ${connection}. Ajoutez un bot de cette connexion, ou passez sur ${connection} pour créer un groupe avec ses bots.`
  },
  de: {
    gatewayPeer: (owner, local = false) => `Läuft auf ${local ? 'diesem Gerät' : owner}, auch wenn dieser Desktop geschlossen ist. Auf anderen Computern empfangen Bots nur Text.`,
    startGatewayGroup: 'Gruppenchat starten, der ohne Desktop läuft',
    upgradeHistory: 'Frühere Nachrichten bleiben in diesem Gruppenchat.',
    pickBots: max => `Wählen Sie 2–${max} Bots.`,
    desktopMembership: 'Lokale Mitgliedschaften werden über das jeweilige Bot-Profil synchronisiert; Mitglieder anderer Computer bleiben auf diesen Gruppenchat beschränkt.',
    gatewayMembership: 'Die Mitglieder werden mit diesem Gruppenchat auf seinem Computer gespeichert.',
    checkingPeers: 'Es wird geprüft, ob diese Computer den Gruppenchat ohne Desktop ausführen können…',
    peerCheckFailed: (connection, local = false) => `${local ? 'Dieses Gerät' : connection} konnte nicht geprüft werden. Versuchen Sie es erneut, um zu prüfen, ob dieser Gruppenchat ohne Desktop laufen kann.`,
    desktopDriven:
      'Läuft, solange dieser Desktop geöffnet ist: Bots auf anderen Rechnern antworten über ihr eigenes Gateway, und dieser Desktop leitet die Nachrichten zwischen ihnen weiter.',
    namedProfileFallback: 'Bots aus einem benannten Profil auf einem anderen Computer können einem Gruppenchat, der ohne Desktop läuft, noch nicht beitreten.',
    groupsOffline: (connection, bot) => `${connection} ist offline – verbinde es neu, um die Gruppen von ${bot} zu ändern.`,
    memberOffline: (connection, bot) => `${connection} ist offline – verbinde es neu, um ${bot} zu entfernen.`,
    memberSaveFailed: (connection, bot) => `Mitglieder nicht gespeichert: ${connection} hat die Änderung für ${bot} nicht übernommen. Nichts wurde geändert; versuche es erneut, sobald es erreichbar ist.`,
    memberRollbackFailed: (connection, bot) =>
      `Mitglieder nicht gespeichert, und die bereits vorgenommene Änderung für ${bot} auf ${connection} ließ sich nicht rückgängig machen: ${connection} hat sie möglicherweise noch. Der Gruppenchat zeigt den zuletzt von ${connection} gemeldeten Stand; verbinde es neu und prüfe die Mitglieder.`,
    selectionElsewhere: connection => `Diese Bots liegen alle auf ${connection}. Füge einen Bot dieser Verbindung hinzu oder wechsle zu ${connection}, um eine Gruppe mit dessen Bots zu starten.`
  },
  es: {
    gatewayPeer: (owner, local = false) => `Funciona en ${local ? 'este dispositivo' : owner}, incluso cuando este Desktop está cerrado. Los bots de otros equipos solo reciben texto.`,
    startGatewayGroup: 'Crear un chat de grupo que funcione sin Desktop',
    upgradeHistory: 'Los mensajes anteriores permanecen en este chat de grupo.',
    pickBots: max => `Elige de 2 a ${max} bots.`,
    desktopMembership: 'Los miembros locales se sincronizan mediante el perfil de cada bot; los de otros equipos quedan vinculados a este chat de grupo.',
    gatewayMembership: 'Los miembros se guardan con este chat de grupo en el equipo que lo ejecuta.',
    checkingPeers: 'Comprobando si estos equipos pueden ejecutar el chat de grupo sin Desktop…',
    peerCheckFailed: (connection, local = false) => `No se pudo comprobar ${local ? 'este dispositivo' : connection}. Reintenta para saber si este chat de grupo puede funcionar sin Desktop.`,
    desktopDriven:
      'Funciona mientras este Desktop esté abierto: los bots de otras máquinas responden desde su propia pasarela y este Desktop transmite los mensajes entre ellos.',
    namedProfileFallback: 'Los bots de un perfil con nombre en otro ordenador todavía no pueden unirse a un chat de grupo que funcione sin Desktop.',
    groupsOffline: (connection, bot) => `${connection} está sin conexión: vuelve a conectarlo para cambiar los grupos de ${bot}.`,
    memberOffline: (connection, bot) => `${connection} está sin conexión: vuelve a conectarlo para quitar a ${bot}.`,
    memberSaveFailed: (connection, bot) => `Miembros no guardados: ${connection} no aceptó el cambio de ${bot}. No se cambió nada; inténtalo de nuevo cuando esté disponible.`,
    memberRollbackFailed: (connection, bot) =>
      `Miembros no guardados, y el cambio ya hecho para ${bot} en ${connection} no se pudo deshacer: puede que ${connection} todavía lo tenga. El chat de grupo muestra lo último que informó ${connection}; vuelve a conectarlo y revisa los miembros.`,
    selectionElsewhere: connection => `Todos estos bots están en ${connection}. Añade un bot de esta conexión o cambia a ${connection} para crear un grupo con sus bots.`
  }
} satisfies Record<string, CrossGatewayGroupMessages>
