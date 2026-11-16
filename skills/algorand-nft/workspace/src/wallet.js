export class WalletController {
  constructor(Connector, validAddress, changed=()=>{}) {
    this.Connector=Connector; this.validAddress=validAddress; this.changed=changed;
    this.accounts=[]; this.selected=''; this.error=''; this.busy=false; this.unsub=[]; this.generation=0;
  }
  reset() {
    this.generation++;
    for (const unsubscribe of this.unsub) unsubscribe();
    this.unsub=[]; this.instance?.dispose(); this.instance=null;
    this.accounts=[]; this.selected=''; this.busy=false; this.changed();
  }
  async connect() {
    if (this.busy || this.instance) return;
    this.error=''; this.busy=true;
    const generation=++this.generation;
    try {
      const instance=new this.Connector({chainId:416001, shouldShowSignTxnToast:false});
      this.instance=instance;
      this.unsub.push(instance.on('disconnect',()=>this.reset()));
      this.unsub.push(instance.on('networkChanged',()=>{this.error='Wallet network changed. Connection cleared; explicitly reconnect for MainNet.'; this.reset();}));
      // No await before connect: preserve the explicit button user gesture.
      const result=instance.connect();
      this.changed();
      const accounts=await result;
      if (generation!==this.generation) return;
      if (!Array.isArray(accounts) || !accounts.length || !accounts.every(a=>typeof a==='string' && this.validAddress(a))) throw Error('Wallet returned invalid public addresses.');
      this.accounts=[...new Set(accounts)]; this.selected=this.accounts[0];
    } catch (error) {
      if (generation!==this.generation) return;
      this.error=error?.data?.type || error?.message || String(error);
      this.reset();
    } finally { if (generation===this.generation) {this.busy=false; this.changed();} }
  }
  select(address) {
    if (!this.accounts.includes(address)) throw Error('Select an account returned by this connection.');
    this.selected=address; this.changed();
  }
  async disconnect() {
    const instance=this.instance;
    this.error='';
    // Clear local state immediately, but preserve the live SDK until its
    // disconnect call finishes: dispose() detaches SDK transports.
    this.instance=null; this.reset(); this.busy=true; this.changed();
    try { await instance?.disconnect(); }
    catch (error) {this.error=`Local state cleared; wallet disconnect failed: ${error?.message || error}`;}
    finally {instance?.dispose(); this.busy=false; this.changed();}
  }
}
