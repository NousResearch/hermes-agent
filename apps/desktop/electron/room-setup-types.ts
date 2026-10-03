/** Credential-free IPC intent shared by preload/renderer and Electron main.
 * Keep this module free of runtime imports and private journal/credential types. */
export interface SetupRoute {
  connectionId: string
  profile: string
}

export interface RoomSetupMember {
  member_id: string
  handle: string
  profile: string
  display_name?: string
  connectionId: string
}

export interface RoomSetupInput {
  home: SetupRoute
  name: string
  members: RoomSetupMember[]
  /** Your other computers both allow and are designated to continue the group if its host goes offline. */
  successor?: boolean
}

/** Keep a full copy of an existing group on another of your computers, without adding a Bot. */
export interface RoomBackupInput {
  home: SetupRoute
  roomId: string
  backup: SetupRoute
  successor: boolean
}
