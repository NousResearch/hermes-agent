import {PeraWalletConnect} from '@perawallet/connect';
import {isValidAddress} from 'algosdk';
import {mount} from './app.js';

import {createServices} from './services.js';
import './style.css';

// SDK code is loaded before clicks; no connector is constructed during mount.
mount(document, {Connector:PeraWalletConnect, validAddress:isValidAddress,
  mintServices:createServices(),
  storage:{getItem:key=>localStorage.getItem(key),setItem:(key,value)=>localStorage.setItem(key,value),removeItem:key=>localStorage.removeItem(key)}});
