import {Algodv2} from 'algosdk';

export const NODE='https://mainnet-api.algonode.cloud';
// Construction creates no requests. Each method is one finite SDK request;
// in particular a POST is never retried, even after a timeout.
export function createServices({Client=Algodv2}={}) {
 const client=new Client('',NODE,'');
 async function run(request) {
  const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),15000);
  try{return await request.do(undefined,{signal:controller.signal,credentials:'omit',referrerPolicy:'no-referrer',redirect:'error'});}
  finally{clearTimeout(timer);}
 }
 return {
  params:()=>run(client.getTransactionParams()),
  account:address=>run(client.accountInformation(address)),

  send:bytes=>run(client.sendRawTransaction(bytes)),
  pending:txid=>run(client.pendingTransactionInformation(txid)),
  asset:id=>run(client.getAssetByID(id))
 };
}
